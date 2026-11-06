"""Read-only PCM WAV audit, using only the Python standard library.

Fixed diagnostic thresholds (not trained or selected on public_test):
too_short: duration < 0.25 s; silence: abs(amplitude) <= -40 dBFS;
mostly_silent: silence_ratio >= 0.95; clipping: abs(amplitude) >= 0.999;
clipped: clip_ratio >= 0.001. Ratios count individual PCM samples.
Only uncompressed 16-bit PCM is measured; other formats get bad_header.
No transcript or context is used. turn_index is zero-based in full history.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate_speech.io_utils import speech_work_dir

THRESHOLDS = dict(too_short_s=0.25, silence_dbfs=-40, mostly_silent_ratio=0.95,
                  clip_amplitude=0.999, clip_ratio=0.001)
EXPECTED = {'train': {'current': 1545, 'history': 3491},
            'public_test': {'current': 634, 'history': 1299}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def relative_audio(raw, split_dir):
    if not isinstance(raw, str) or not raw:
        raise ValueError('Missing or invalid audio path')
    rel = PurePosixPath(raw.replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts or ':' in raw:
        raise ValueError(f'Unsafe audio path: {raw!r}')
    path = (split_dir / Path(*rel.parts)).resolve()
    if not path.is_relative_to(split_dir.resolve()):
        raise ValueError(f'Audio escapes split: {raw!r}')
    return path.relative_to(split_dir.resolve()).as_posix()


def references(split_dir, split):
    refs = []
    ids = set()
    with (split_dir / f'{split}.jsonl').open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                sid = row['id']
                if not isinstance(sid, str) or sid in ids:
                    raise ValueError('Invalid or duplicate sample id')
                ids.add(sid)
                turns = [(None, row)] + [(i, t) for i, t in enumerate(row.get('history', []))
                                         if t.get('role') == 'user']
                for index, turn in turns:
                    refs.append(dict(split=split, sample_id=sid,
                                     kind='current' if index is None else 'history',
                                     turn_index=index,
                                     audio_path=relative_audio(turn.get('audio'), split_dir)))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'{split} JSONL line {line_no}: {exc}') from exc
    return refs, ids


def inspect_audio(path):
    result = dict(exists=path.is_file(), sample_rate=None, channels=None, bit_depth=None,
                  duration_s=None, rms_dbfs=None, silence_ratio=None, clip_ratio=None, flags=[])
    if not result['exists']:
        result['flags'].append('missing')
        result['error'] = 'File does not exist or is not a regular file'
        return result
    try:
        with wave.open(str(path), 'rb') as wav:
            rate, channels, width, frames = (wav.getframerate(), wav.getnchannels(),
                                            wav.getsampwidth(), wav.getnframes())
            result.update(sample_rate=rate, channels=channels, bit_depth=width * 8)
            if rate <= 0 or channels <= 0 or width != 2 or wav.getcomptype() != 'NONE':
                raise ValueError('Expected uncompressed 16-bit PCM with positive rate/channels')
            result['duration_s'] = frames / rate
            count = silent = clipped = squares = frames_read = 0
            silence_level = 32768 * 10 ** (THRESHOLDS['silence_dbfs'] / 20)
            clip_level = 32768 * THRESHOLDS['clip_amplitude']
            while True:
                data = wav.readframes(65536)
                if not data:
                    break
                if len(data) % (channels * width):
                    raise ValueError('Incomplete PCM frame')
                frames_read += len(data) // (channels * width)
                values = array('h', data)
                if sys.byteorder != 'little':
                    values.byteswap()
                for value in values:
                    amp = abs(value)
                    squares += value * value
                    silent += amp <= silence_level
                    clipped += amp >= clip_level
                count += len(values)
            if frames_read != frames:
                raise ValueError(f'Truncated data: expected {frames}, read {frames_read} frames')
            if not count:
                raise ValueError('Empty audio data')
            result.update(rms_dbfs=20 * math.log10(math.sqrt(squares / count) / 32768) if squares else None,
                          silence_ratio=silent / count, clip_ratio=clipped / count)
            if result['duration_s'] < THRESHOLDS['too_short_s']:
                result['flags'].append('too_short')
            if result['silence_ratio'] >= THRESHOLDS['mostly_silent_ratio']:
                result['flags'].append('mostly_silent')
            if result['clip_ratio'] >= THRESHOLDS['clip_ratio']:
                result['flags'].append('clipped')
    except (OSError, EOFError, ValueError, wave.Error) as exc:
        result['flags'].append('bad_header')
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def percentiles(values):
    values = sorted(values)
    def percentile(q):
        if not values:
            return None
        pos = (len(values) - 1) * q
        low = int(pos)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (pos - low)
    return dict(p50=percentile(0.5), p95=percentile(0.95), max=max(values) if values else None)


def summarize(rows):
    durations = [r['duration_s'] for r in rows if r['duration_s'] is not None and 'bad_header' not in r['flags']]
    totals = defaultdict(float)
    invalid_samples = set()
    for row in rows:
        sid = row['sample_id']
        if sid is None:
            continue
        if row['duration_s'] is None or 'bad_header' in row['flags']:
            invalid_samples.add(sid)
        else:
            totals[sid] += row['duration_s']
    return dict(file_count=len(rows), valid_duration_count=len(durations), total_hours=sum(durations) / 3600,
                duration_s=percentiles(durations),
                sample_total_duration_s=percentiles([v for k, v in totals.items() if k not in invalid_samples]),
                incomplete_sample_count=len(invalid_samples),
                flag_counts=dict(Counter(flag for row in rows for flag in row['flags'])),
                missing_count=sum('missing' in r['flags'] for r in rows),
                unreferenced_count=sum(not r['referenced'] for r in rows))


def run(data_root, splits, out, limit=None):
    data_root, out = data_root.resolve(), out.resolve()
    work = speech_work_dir()
    if not out.is_relative_to(work.resolve()):
        raise ValueError(f'Output must be inside {work}')
    if out.is_relative_to(data_root) or data_root.is_relative_to(out):
        raise ValueError('Output and DATA_ROOT must not overlap')
    if limit is not None and limit <= 0:
        raise ValueError('--limit must be positive')
    started = time.perf_counter()
    rows, split_stats = [], {}
    for split in splits:
        split_dir = data_root / split
        refs, ids = references(split_dir, split)
        all_paths = {r['audio_path'] for r in refs}
        disk_paths = {p.relative_to(split_dir).as_posix() for p in (split_dir / 'audio').rglob('*')
                      if p.is_file() and p.suffix.lower() == '.wav'}
        selected_ids = list(dict.fromkeys(r['sample_id'] for r in refs))[:limit]
        selected_id_set = set(selected_ids)
        selected = [r for r in refs if r['sample_id'] in selected_id_set]
        counts = dict(Counter(r['kind'] for r in refs))
        grouped = defaultdict(list)
        for ref in selected:
            grouped[ref['audio_path']].append(ref)
        # Preview orphans are limited to directories containing selected referenced files.
        selected_dirs = {str(PurePosixPath(p).parent) for p in grouped}
        orphans = disk_paths - all_paths
        for path in sorted(orphans):
            if limit is None or str(PurePosixPath(path).parent) in selected_dirs:
                grouped[path] = []
        for file_index, (path, links) in enumerate(grouped.items(), 1):
            row = dict(links[0]) if links else dict(split=split, sample_id=None, kind=None,
                                                   turn_index=None, audio_path=path)
            row.update(referenced=bool(links), references=links, **inspect_audio(split_dir / path))
            if not links:
                row['flags'].append('unreferenced')
            rows.append(row)
            if file_index % 250 == 0 or file_index == len(grouped):
                print(f'{split}: {file_index}/{len(grouped)} files audited', file=sys.stderr, flush=True)
        local = [r for r in rows if r['split'] == split]
        split_stats[split] = dict(sample_count=len(ids), selected_sample_count=len(selected_ids),
                                 reference_counts=counts, expected_reference_counts=EXPECTED[split],
                                 reference_counts_match=counts == EXPECTED[split],
                                 inventory_wav_count=len(disk_paths),
                                 inventory_unreferenced_count=len(orphans),
                                 inventory_missing_count=len(all_paths - disk_paths),
                                 duplicate_reference_count=len(refs) - len(all_paths),
                                 all=summarize(local),
                                 by_kind={kind: summarize([r for r in local if r['kind'] == kind])
                                          for kind in ('current', 'history', None)})
    summary = dict(mode='preview' if limit else 'full', limit_samples_per_split=limit,
                   thresholds=THRESHOLDS, rms_zero_convention='null means -infinity when valid and entirely silent',
                   turn_index_convention='zero-based index into full history array',
                   file_count=len(rows), splits=split_stats,
                   elapsed_s=time.perf_counter() - started,
                   expected_full_wav_count=6969,
                   inventory_wav_count=sum(s['inventory_wav_count'] for s in split_stats.values()))
    if set(splits) == set(EXPECTED):
        summary['inventory_wav_count_match'] = summary['inventory_wav_count'] == 6969
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'audio_audit.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(out / 'audio_audit_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--splits', nargs='+', choices=list(EXPECTED), default=list(EXPECTED))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, help='Preview first N samples per split; inventory scans all filenames')
    args = parser.parse_args()
    if len(set(args.splits)) != len(args.splits):
        parser.error('Duplicate splits')
    run(args.data_root, args.splits, args.out, args.limit)


if __name__ == '__main__':
    main()
