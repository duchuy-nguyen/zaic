"""Transcribe current + all user-history audio. Never consumes gold text.

--no-cache disables BOTH reads and writes. Cache timing runs are not live timing.
Resume reuses only successful rows with matching audio hash and config cache key.
Writes each completed mini-batch with flush/fsync. CPU timing is CPU-only evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate_speech.asr import MODEL_ID, REVISION, SEED, PhoWhisperASR, decoding_config, model_weight_provenance, package_versions
from lifemate_speech.cache import TranscriptCache, cache_key, config_hash
from lifemate_speech.io_utils import atomic_json, audio_sha256, ensure_work_path, load_audio_refs, ref_key, speech_work_dir, wav_info
from lifemate_speech.normalize import NORMALIZATION_VERSION, normalize


def code_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(list((root/'src').rglob('*.py')) + list((root/'scripts').glob('*.py'))):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_resume(path):
    """Only a crash-truncated final line may be discarded, never a corrupt middle line."""
    lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    result, seen = [], set()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if i == len(lines)-1 and not line.endswith(b'\n'):
                break
            raise ValueError(f'Corrupt resume JSONL at line {i+1}')
        key = ref_key(row)
        if key in seen:
            raise ValueError(f'Duplicate resume row: {key}')
        seen.add(key)
        result.append(row)
    return result


def execute(args, provider_factory=PhoWhisperASR):
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    out = ensure_work_path(args.out)
    cache_dir = ensure_work_path(args.cache_dir)
    model_cache = ensure_work_path(args.model_cache_dir)
    data_root = args.data_root.resolve()
    for target in (out, cache_dir, model_cache):
        if target.is_relative_to(data_root) or data_root.is_relative_to(target):
            raise ValueError('Outputs/cache must not overlap DATA_ROOT')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    config = decoding_config(args.device, batch_size=args.batch_size)
    refs = load_audio_refs(data_root, args.split, args.limit, getattr(args, 'sample_ids', None))
    if not refs:
        raise ValueError('No audio references selected')
    transcript_path = out / 'transcripts.jsonl'
    if not args.resume and transcript_path.exists():
        raise ValueError('Output already exists; use --resume or another --out')
    selected_identity = [{k: r[k] for k in ('sample_id', 'split', 'kind', 'turn_index', 'audio_path')} for r in refs]
    fingerprint = config_hash(dict(config=config, refs=selected_identity, normalization=NORMALIZATION_VERSION))
    old_manifest = None
    if args.resume and (out/'run_manifest.json').exists():
        old_manifest = json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
        if old_manifest.get('selection_config_hash') != fingerprint:
            raise ValueError('Resume selection/config differs from existing run')
    old = {ref_key(r): r for r in read_resume(transcript_path)} if args.resume else {}
    if set(old) - {ref_key(r) for r in refs}:
        raise ValueError('Resume contains unknown references')
    records, pending = [], []
    for ref in refs:
        row = {k: v for k, v in ref.items() if k not in ('path', 'input_error')}
        row.update(audio_sha256=None, text_raw=None, text_norm=None, asr_latency_s=None,
                   audio_duration_s=None, from_cache=False, asr_model=MODEL_ID, asr_revision=REVISION,
                   decoding_config=config, normalization_version=NORMALIZATION_VERSION, error=None,
                   flags=[], batch_size=0, batch_wall_time_s=None, batch_id=None,
                   latency_method='equal_batch_wall_time_allocation', cache_key=None, run_id=run_id)
        try:
            if ref['input_error']:
                raise ValueError(ref['input_error'])
            row['audio_sha256'] = audio_sha256(ref['path'])
            row['cache_key'] = cache_key(row['audio_sha256'], REVISION, config)
            info = wav_info(ref['path'])
            row['audio_duration_s'] = info['audio_duration_s']
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            if 'AudioTooLong' in row['error']:
                row['flags'].append('audio_too_long')
            else:
                row['flags'].append('audio_input_error')
        previous = old.get(ref_key(ref))
        if previous and previous.get('error') is None and previous.get('cache_key') == row['cache_key'] and previous.get('audio_path') == row['audio_path'] and not row['error']:
            records.append(dict(previous, resumed=True))
        else:
            row['resumed'] = False
            pending.append((ref, row))
    cache = TranscriptCache(cache_dir)
    batches, warnings = [], []
    manifest = dict(run_id=run_id, status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    split=args.split, limit=args.limit, device=args.device, batch_size=args.batch_size,
                    cache_mode='disabled_read_and_write' if args.no_cache else 'enabled',
                    cache_dir=str(cache_dir), model_cache_dir=str(model_cache), local_files_only=args.local_files_only,
                    selection_config_hash=fingerprint, code_sha256=code_hash(), libraries=package_versions(),
                    argv=list(sys.argv), executable=sys.executable,
                    model_weights=model_weight_provenance(model_cache),
                    selected_sample_ids=list(dict.fromkeys(r['sample_id'] for r in refs)),
                    python=platform.python_version(), platform=platform.platform(), seed=SEED,
                    asr_model=MODEL_ID, asr_revision=REVISION, decoding_config=config,
                    normalization_version=NORMALIZATION_VERSION,
                    latency_method='equal allocation of batch wall (read/features/transfer/generate/decode); model load excluded',
                    latency_warning=('CPU timing cannot establish competition latency. ' if args.device == 'cpu' else 'Target-hardware end-to-end validation still required. ') + 'Cached/resumed rows are not live timing. Cross-sample batches: allocated effort is not sample wall time.',
                    selected_audio_count=len(refs), selected_sample_count=len({r['sample_id'] for r in refs}),
                    resumed_count=len(records), previous_run_id=old_manifest.get('run_id') if old_manifest else None,
                    gpu_driver=None, batches=batches, warnings=warnings)
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        manifest['gpu_driver'] = proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'run_manifest.json', manifest)
    provider = None
    # Atomically retain successes before append; a crash during rewrite cannot lose old successes.
    prepared = transcript_path.with_name('transcripts.prepared.tmp')
    with prepared.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    prepared.replace(transcript_path)
    with transcript_path.open('a', encoding='utf-8') as stream:
        def save(row):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            misses = []
            for ref, row in pending:
                if row['error']:
                    records.append(row)
                    save(row)
                    continue
                cached = None
                if not args.no_cache:
                    cache_started = time.perf_counter()
                    try:
                        cached = cache.get(row['cache_key'])
                        if cached is not None and (cached.get('audio_sha256') != row['audio_sha256'] or cached.get('asr_revision') != REVISION or cached.get('decoding_config') != config):
                            raise ValueError('Cached provenance mismatch')
                    except (OSError, ValueError, KeyError) as exc:
                        cached = None
                        warnings.append(dict(audio_path=row['audio_path'], cache_error=str(exc)))
                    row['cache_read_latency_s'] = time.perf_counter()-cache_started
                if cached is not None:
                    row.update(text_raw=cached['text_raw'], text_norm=normalize(cached['text_raw']), from_cache=True,
                               asr_latency_s=0.0, batch_wall_time_s=0.0,
                               cached_source_timing={k: cached.get(k) for k in ('run_id', 'asr_latency_s', 'batch_id', 'batch_size', 'batch_wall_time_s')})
                    records.append(row)
                    save(row)
                else:
                    misses.append((ref, row))
            misses.sort(key=lambda pair: pair[1]['audio_duration_s'])
            for offset in range(0, len(misses), args.batch_size):
                batch = misses[offset:offset+args.batch_size]
                if provider is None:
                    provider = provider_factory(device=args.device, batch_size=args.batch_size,
                                                model_cache_dir=str(model_cache), local_files_only=args.local_files_only)
                results = provider.transcribe_batch([ref['path'] for ref, _ in batch])
                if len(results) != len(batch):
                    raise RuntimeError('ASR result count mismatch')
                batch_id = f'{run_id}:{offset//args.batch_size}'
                batches.append(dict(batch_id=batch_id, actual_batch_size=len(batch),
                                    batch_wall_time_s=results[0]['batch_wall_time_s'],
                                    audio_refs=[list(ref_key(row)) for _, row in batch]))
                for (_, row), result in zip(batch, results, strict=True):
                    row.update(text_raw=result['text_raw'], text_norm=normalize(result['text_raw']) if result['error'] is None else None,
                               asr_latency_s=result['latency_s'], batch_size=result['batch_size'],
                               batch_wall_time_s=result['batch_wall_time_s'], batch_id=batch_id, error=result['error'])
                    if row['error'] is None and not args.no_cache:
                        try:
                            cache.put(row['cache_key'], row)
                        except OSError as exc:
                            warnings.append(dict(audio_path=row['audio_path'], cache_write_error=str(exc)))
                    records.append(row)
                    save(row)
                print(f"{len(records)}/{len(refs)} audio records saved", file=sys.stderr, flush=True)
            manifest['status'] = 'completed' if all(r['error'] is None for r in records) else 'completed_with_errors'
        except BaseException as exc:
            manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            manifest['fatal_error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            errors = [r for r in records if r['error'] is not None]
            (out/'errors.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in errors), encoding='utf-8')
            manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), total_wall_time_s=time.perf_counter()-started,
                            recorded_audio_count=len(records), success_count=sum(r['error'] is None for r in records),
                            error_count=len(errors), unprocessed_count=len(refs)-len(records),
                            from_cache_count=sum(r['from_cache'] for r in records),
                            fresh_success_count=sum(r['error'] is None and not r['from_cache'] and not r.get('resumed') for r in records),
                            model_initialized=provider is not None and provider.model is not None,
                            model_load_s=provider.model_load_s if provider else 0.0,
                            effective_generation_config=provider.effective_generation_config if provider else None,
                            hardware=provider.hardware if provider else {})
            manifest['model_weights'] = model_weight_provenance(model_cache)
            atomic_json(out/'run_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    work = speech_work_dir()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--split', choices=('train', 'public_test'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cache-dir', type=Path, default=work/'transcript_cache')
    parser.add_argument('--model-cache-dir', type=Path, default=work/'hf_models')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample-ids', nargs='+', help='Explicit train/sample IDs for targeted smoke; all user history included')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = execute(args)
    print(json.dumps({k: result[k] for k in ('status', 'selected_audio_count', 'success_count', 'error_count', 'from_cache_count', 'total_wall_time_s')}, indent=2))
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
