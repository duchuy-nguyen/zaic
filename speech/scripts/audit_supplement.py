"""Read existing audit and hash WAV bytes, without selecting ASR rules on test data."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifemate_speech.io_utils import speech_work_dir


def percentile(values, q, method='linear'):
    values = sorted(values)
    if not values:
        return None
    if method == 'nearest-rank':
        return values[max(0, math.ceil(q * len(values)) - 1)]
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if method == 'lower':
        return values[lo]
    if method == 'higher':
        return values[hi]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    work = speech_work_dir()
    if not args.out.resolve().is_relative_to(work.resolve()):
        parser.error('Output must be in speech_work')
    rows = [json.loads(line) for line in args.audit.read_text(encoding='utf-8').splitlines()]
    totals, counts = defaultdict(float), Counter()
    hashes, errors, hash_rows = defaultdict(list), [], []
    for r in rows:
        if r['referenced']:
            counts[r['split'], r['sample_id']] += 1
            if r['duration_s'] is not None:
                totals[r['split'], r['sample_id']] += r['duration_s']
        path = args.data_root / r['split'] / r['audio_path']
        meta = {k: r[k] for k in ('split', 'sample_id', 'kind', 'turn_index', 'audio_path')}
        try:
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            hashes[sha].append(meta)
            hash_rows.append(dict(meta, audio_sha256=sha, error=None))
        except OSError as exc:
            errors.append(dict(meta, error=str(exc)))
            hash_rows.append(dict(meta, audio_sha256=None, error=str(exc)))
    stats = {}
    for split, target in [('train', 41.081), ('public_test', 39.633)]:
        durations = [v for (s, _), v in totals.items() if s == split]
        number = [v for (s, _), v in counts.items() if s == split]
        methods = {m: percentile(durations, .95, m) for m in ('nearest-rank', 'higher', 'lower', 'linear')}
        rms = {}
        for kind in ('all', 'current', 'history'):
            group = [r for r in rows if r['split'] == split and (kind == 'all' or r['kind'] == kind)]
            vals = [r['rms_dbfs'] for r in group if r['rms_dbfs'] is not None]
            rms[kind] = dict(total_files=len(group), finite_rms_files=len(vals), null_rms_files=len(group)-len(vals),
                             min=min(vals) if vals else None, max=max(vals) if vals else None,
                             **{name: percentile(vals, q) for name, q in [('p1', .01), ('p50', .5), ('p99', .99)]})
            ranked_group = sorted([r for r in group if r['rms_dbfs'] is not None], key=lambda r: r['rms_dbfs'])
            rms[kind]['lowest_10'] = [dict(audio_path=r['audio_path'], rms_dbfs=r['rms_dbfs']) for r in ranked_group[:10]]
            rms[kind]['highest_10'] = [dict(audio_path=r['audio_path'], rms_dbfs=r['rms_dbfs']) for r in reversed(ranked_group[-10:])]
        dup = [[r for r in refs if r['split'] == split] for refs in hashes.values()]
        dup = [refs for refs in dup if len(refs) > 1]
        stats[split] = dict(p95_sample_duration_s=methods, plan_p95_s=target,
                           matching_methods_to_3_decimals=[m for m, v in methods.items() if round(v, 3) == target],
                           files_per_sample=dict(min=min(number), p50=percentile(number, .5), p95=percentile(number, .95), max=max(number), histogram=dict(sorted(Counter(number).items()))),
                           rms_dbfs=rms,
                           duplicates_within=dict(groups=len(dup), files_in_groups=sum(map(len, dup)), redundant_files=sum(len(g)-1 for g in dup), pairs=sum(len(g)*(len(g)-1)//2 for g in dup)))
    cross = [refs for refs in hashes.values() if len({r['split'] for r in refs}) > 1]
    duplicate_groups = [dict(audio_sha256=sha, files=refs) for sha, refs in hashes.items() if len(refs) > 1]
    ranked = sorted([r for r in rows if r['rms_dbfs'] is not None], key=lambda r: r['rms_dbfs'])
    extreme = lambda rs: [{k: r[k] for k in ('split', 'sample_id', 'kind', 'turn_index', 'audio_path', 'rms_dbfs', 'flags')} for r in rs]
    result = dict(splits=stats, hash_file_count=len(hash_rows), hash_success_count=sum(r['audio_sha256'] is not None for r in hash_rows),
                  unique_sha256_count=len(hashes), duplicate_groups=len(duplicate_groups),
                  redundant_files=sum(len(g['files'])-1 for g in duplicate_groups),
                  cross_split=dict(groups=len(cross), files=sum(map(len, cross)),
                                   train_public_pairs=sum(sum(r['split']=='train' for r in g)*sum(r['split']=='public_test' for r in g) for g in cross)),
                  rms_lowest_10=extreme(ranked[:10]), rms_highest_10=extreme(list(reversed(ranked[-10:]))),
                  errors=errors, rules='Physical paths counted once. Byte SHA256, not acoustic near-duplicate. RMS quantiles linear; null excluded and counted.')
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in [('audit_supplement.json', result)]:
        (args.out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, values in [('audio_hashes.jsonl', hash_rows), ('duplicate_groups.jsonl', duplicate_groups)]:
        (args.out/name).write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('hash_file_count', 'unique_sha256_count', 'duplicate_groups', 'cross_split')}, indent=2))


if __name__ == '__main__':
    main()
