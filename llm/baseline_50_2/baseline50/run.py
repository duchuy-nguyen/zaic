"""Live current/history ASR and unchanged V2 tool actor; no uploads or training."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

from . import actor
from .asr import MediumASR, audio_paths, audio_view
from .backend import LocalServer
from .io import (read, sha, write_json, new_text, acquire_lock, load_config,
                 verify_assets, load_inputs, resume_records)
from .scoring import score
from .policy import words, distance

SUCCESS = {'finished', 'asked_missing_info', 'call_budget_exhausted'}


def audit(records, rows, tools, schema_validate):
    if [r['id'] for r in records] != [r['id'] for r in rows]:
        raise ValueError('Output coverage/order mismatch')
    flags = {k: [] for k in ('reasoning', 'token_context', 'output_truncation', 'asr_missing', 'asr_hash', 'forbidden_or_invalid_tools')}
    for r, row in zip(records, rows):
        i = r['id']
        allowed = {n for a in row['context']['installed_apps'].values() for n in a['tools']} | {'sys.ask_missing_info'}
        if any(not t['ok'] for t in schema_validate(r['tool_calls'], tools, allowed)):
            flags['forbidden_or_invalid_tools'].append(i)
        actual = [{k: t['proposal'][k] for k in ('name', 'arguments')} for t in r['trace'] if t.get('executed')]
        if actual != r['tool_calls']:
            raise ValueError('Executed trace differs from ToolEnv call log: ' + i)
        for t in r['trace']:
            choice = t['response']['choices'][0]; message = choice['message']
            if t['prompt_tokens_rendered'] != t['response'].get('usage', {}).get('prompt_tokens') or t['prompt_tokens_rendered'] + 512 > 16384:
                flags['token_context'].append(i)
            if message.get('reasoning_content') or '<think>' in (message.get('content') or ''):
                flags['reasoning'].append(i)
            if choice.get('finish_reason') == 'length':
                flags['output_truncation'].append(i)
        a = r.get('asr')
        if not a or set(a['transcripts']) != set(audio_paths(row)):
            flags['asr_missing'].append(i)
        elif hashlib.sha256(json.dumps(a['transcripts'], sort_keys=True, ensure_ascii=False).encode()).hexdigest() != a['transcript_sha256']:
            flags['asr_hash'].append(i)
    return {k: sorted(set(v)) for k, v in flags.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'val', 'public'])
    parser.add_argument('--config', type=Path, default=Path('config.local.json'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); config = load_config(args.config)
    assets = verify_assets(config)
    if args.mode == 'check':
        for mode in ('val', 'public'):
            rows, _ = load_inputs(config, mode, assets)
            print('VERIFIED', mode, len(rows))
        print('ASSETS_VERIFIED; no model inference, ToolEnv replay, training or upload.'); return
    output = config['output_dir'] / args.mode
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output / 'run.lock'); state = None
    try:
        rows, tools = load_inputs(config, args.mode, assets); ids = [r['id'] for r in rows]
        split_name = 'train' if args.mode == 'val' else 'public_test'
        source = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
        frozen = {'release': assets['release'], 'mode': args.mode, 'ids': ids, 'paths': {k: str(v) for k, v in config.items()},
                  'assets_manifest_sha256': sha(Path(__file__).parent / 'assets.manifest.json'), 'source_sha256': source,
                  'policy': {'context': 16384, 'max_output': 512, 'calls': 8, 'repairs': 2, 'seconds_live_asr_included': 30,
                             'thinking': False, 'temperature': 0, 'seed': 20261103}}
        marker = output / 'run_manifest.json'
        if marker.exists():
            state = read(marker)
            if state['configuration'] != frozen:
                raise ValueError('Existing output belongs to different code/assets/config; use a NEW output folder')
            if state['status'] == 'completed':
                print('ALREADY_COMPLETED', output); return
            if not args.resume:
                raise ValueError('Interrupted run requires explicit --resume; no automatic retries')
        else:
            state = {'status': 'preparing', 'completed_samples': 0, 'configuration': frozen, 'startup_sessions': []}
            write_json(marker, state)
        committed = resume_records(output, ids)
        actor.bind_package(config['assets_dir'] / 'private_data')
        if 'zaic_tools' in sys.modules:
            raise ValueError('Run one split per process; do not mix TRAIN/public SDK modules')
        sys.path.insert(0, str(config['dataset_dir'] / split_name))
        sdk = importlib.import_module('zaic_tools')
        if len(committed) < len(ids):
            asr = MediumASR(config['assets_dir'] / 'models/asr', config['dataset_dir'], split_name, assets['asr_vocab_sha256'])
            asr.warmup(read(config['assets_dir'] / 'private_data/split.json')['train_fit'][0])
            session = len(state['startup_sessions']) + 1
            with LocalServer(config['assets_dir'] / assets['qwen_file'], config['llama_server'], output / f'server-session-{session}') as backend:
                state['startup_sessions'].append({'command': backend.public_command()})
                state['status'] = 'running'; write_json(marker, state)
                for row in rows[len(committed):]:
                    write_json(output / 'inflight.json', {'id': row['id'], 'ordinal': len(committed)})
                    started = time.monotonic()
                    env = sdk.ToolEnv(split_name, row['id'], root=config['dataset_dir']); details = None
                    try:
                        mapping, details = asr.transcribe(audio_paths(row), started + 30)
                        view = audio_view(row, mapping); remaining = 30 - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError('ASR exhausted full sample budget')
                        result = actor.infer(view, tools, env, backend, seconds=remaining)
                    except Exception as error:
                        result = {'id': row['id'], 'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash(), 'trace': [],
                                  'status': 'timeout' if isinstance(error, TimeoutError) else 'asr_error', 'error': repr(error), 'repairs': 0}
                    result['llm_and_agent_s'] = result.pop('elapsed_s', 0.)
                    result['elapsed_s'] = time.monotonic() - started; result['asr'] = details
                    if result['elapsed_s'] > 30:
                        result['status'] = 'timeout'
                    write_json(output / 'records' / (row['id'] + '.json'), result)
                    committed[row['id']] = result; (output / 'inflight.json').unlink()
                    state['completed_samples'] = len(committed); write_json(marker, state)
                    print(f"[{args.mode} {len(committed)}/{len(ids)}] {row['id']} {result['status']} {result['elapsed_s']:.2f}s", flush=True)
                    if backend.process.poll() is not None:
                        raise RuntimeError('Owned llama-server exited')
        records = [committed[i] for i in ids]
        flags = audit(records, rows, tools, sdk.schema_validate)
        new_text(output / 'traces.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        summary = {'n': len(records), 'audit_flags': flags, 'timeouts': sum(r['status'] == 'timeout' for r in records),
                   'failed_samples': sum(r['status'] not in SUCCESS or r['elapsed_s'] > 30 for r in records),
                   'statuses': dict(Counter(r['status'] for r in records)), 'uploaded': False}
        if args.mode == 'public':
            target = output / 'submission_baseline_50_2.jsonl'
            new_text(target, ''.join(json.dumps({k: r[k] for k in ('id', 'tool_calls', 'state_hash')}, ensure_ascii=False, allow_nan=False) + '\n' for r in records))
            if target.stat().st_size >= 5 * 1024**2:
                raise ValueError('Submission exceeds 5MiB')
            summary.update(submission_sha256=sha(target), public_score=None)
            new_text(output / 'SOLUTION_DESCRIPTION.txt', (Path(__file__).parent / 'SOLUTION_DESCRIPTION.txt').read_text(encoding='utf8'))
        else:
            # Labels are opened only after ALL inference is committed, never passed to actor.
            refs = read(config['assets_dir'] / 'private_data/train_references.json')['references']
            scores = {r['id']: score(r, refs[r['id']], tools, failed=r['status'] not in SUCCESS or r['elapsed_s'] > 30) for r in records}
            durations = sorted(r['elapsed_s'] for r in records)
            summary.update(score_100=statistics.mean(s['score_spec_local'] for s in scores.values()),
                           tool_f1_percent=100 * statistics.mean(s['tool_f1'] for s in scores.values()),
                           state_match_percent=100 * statistics.mean(s['state_match'] for s in scores.values()),
                           p50_s=statistics.median(durations), p95_s=durations[math.ceil(.95 * len(durations)) - 1],
                           score_scope='local-spec-v1 approximation; not leaderboard; VAL used for development')
            original = {r['id']: r for r in map(json.loads, (config['dataset_dir'] / 'train/train.jsonl').read_text(encoding='utf8').splitlines())}
            errors = {'current': [0, 0], 'history': [0, 0]}
            for r in records:
                row = original[r['id']]; mapping = (r.get('asr') or {}).get('transcripts', {})
                for kind, turn in [('current', row)] + [('history', t) for t in row['history'] if t['role'] == 'user']:
                    a, b = words(turn['audio_text']), words(mapping.get(turn['audio'], ''))
                    errors[kind][0] += distance(a, b); errors[kind][1] += len(a)
            summary['asr_wer_percent'] = {k: 100 * e / n for k, (e, n) in errors.items()}
            summary['asr_wer_percent']['all'] = 100 * sum(e for e, n in errors.values()) / sum(n for e, n in errors.values())
            write_json(output / 'scores.json', scores)
        write_json(output / 'summary.json', summary)
        state.update(status='completed', completed_samples=len(records), audit_flags=flags); write_json(marker, state)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        if state is not None:
            state.update(status='paused_by_user' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
            write_json(output / 'run_manifest.json', state)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
