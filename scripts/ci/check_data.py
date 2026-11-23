"""Read-only checks of DATA assets committed to git; not official scoring.

Kiểm (lỗi -> exit 1):
  1. Split: data/manifests/split_v1.json nhất quán (FIT/VAL rời nhau, đếm khớp); train_fit_provisional.json == FIT;
     expected_ids_{fit,val}.jsonl (nếu có) khớp split.
  2. Mọi release data/releases/*/release*.json: by_technique cộng = records; các trường *_sha256 đúng dạng;
     nếu release khai split_manifest_sha256 thì phải trùng split_v1.json đang commit.
     Nếu thư mục có synthetic_phase1.jsonl: sha256 file == dataset_sha256, số dòng == records, đếm theo kỹ thuật khớp,
     ID duy nhất, mọi tool có trong catalog.
  3. Manifest bundle OPS (data/releases/ops_train_smoke_v*/manifest.json): ID thuộc TRAIN_FIT, hash đúng dạng.
  4. Không commit data thô nặng trong data/: *.wav, state.json, *.tar/*.gz/*.zip.
Hash tính trên nội dung đã đổi CRLF -> LF (đúng bytes git lưu), nên chạy được cả trên bản checkout Windows.
Cảnh báo (không fail):data/synthetic/synthetic_phase1.jsonl bản cũ còn trên git; *.jsonl ngoài data/releases.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'data'
SHA = re.compile(r'^[0-9a-f]{64}$')
errors, warnings = [], []


def check(cond, msg):
    if not cond:
        errors.append(msg)


def sha(p):
    # Hash the bytes git stores. A Windows checkout with core.autocrlf=true writes text files with CRLF,
    # which would make every recorded LF hash look wrong, so fold CRLF back to LF before hashing.
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]


def main():
    cat_p = ROOT / 'synthetic/synthetic_tools_catalog.json'
    tools = {t['name'] for a in load(cat_p)['apps'] for t in a['tools']} if cat_p.exists() else None

    # 1) split
    split_p = ROOT / 'manifests/split_v1.json'
    split_sha, fit, val = (sha(split_p) if split_p.exists() else None), set(), set()
    check(split_p.exists(), 'Missing data/manifests/split_v1.json')
    if split_p.exists():
        m = load(split_p)
        fit, val = set(m.get('fit', [])), set(m.get('val', []))
        n = m.get('report', {}).get('n', {})
        check(m.get('version') == 'split-v1', 'split_v1.json: version must be split-v1')
        check(fit and val and not fit & val, 'split_v1.json: FIT/VAL must be nonempty and disjoint')
        check((len(fit), len(val), len(fit) + len(val)) == (n.get('fit'), n.get('val'), n.get('total')),
              'split_v1.json: counts do not match report')
        fp = ROOT / 'manifests/train_fit_provisional.json'
        if fp.exists():
            check(set(load(fp)) == fit, 'train_fit_provisional.json must equal split_v1.json["fit"]')
        for name, ids in (('fit', fit), ('val', val)):
            ep = ROOT / f'manifests/expected_ids_{name}.jsonl'
            if ep.exists():
                got = [r.get('id') for r in jsonl(ep)]
                check(len(got) == len(set(got)) and set(got) == ids, f'expected_ids_{name}.jsonl != split_v1 {name}')

    # 2) releases
    for rel_p in sorted(ROOT.glob('releases/*/release*.json')):
        r, tag = load(rel_p), rel_p.parent.name
        check(sum(r.get('by_technique', {}).values()) == r.get('records'), f'{tag}: by_technique sum != records')
        for k, v in r.items():
            if k.endswith('_sha256') and v is not None:
                check(bool(SHA.match(str(v))), f'{tag}: {k} is not sha256')
        if r.get('split_manifest_sha256') and split_sha:
            check(r['split_manifest_sha256'] == split_sha, f'{tag}: split_manifest_sha256 != committed split_v1.json')
        data_p = rel_p.parent / 'synthetic_phase1.jsonl'
        if data_p.exists():
            rows = jsonl(data_p)
            ids = [x['id'] for x in rows]
            check(sha(data_p) == r.get('dataset_sha256'), f'{tag}: synthetic_phase1.jsonl sha256 != dataset_sha256')
            check(len(rows) == r.get('records'), f'{tag}: line count != records')
            check(dict(Counter(x['technique'] for x in rows)) == r.get('by_technique'), f'{tag}: technique counts differ')
            check(len(ids) == len(set(ids)), f'{tag}: duplicate IDs')
            check(not set(r.get('excluded_ids', [])) & set(ids), f'{tag}: excluded IDs still present')
            if tools:
                bad = {c['name'] for x in rows for c in x['label']['tool_calls'] if c['name'] not in tools}
                check(not bad, f'{tag}: tools not in catalog: {sorted(bad)[:5]}')
            if fit:
                check(all(x.get('source_id') in fit for x in rows), f'{tag}: source_id outside TRAIN_FIT')
        val_p = rel_p.parent / rel_p.name.replace('release', 'validation')
        if val_p.exists() and 'error_count' in load(val_p):
            check(load(val_p)['error_count'] == 0, f'{tag}: validation error_count != 0')

    # 3) OPS bundle manifests
    for b_p in sorted(ROOT.glob('releases/ops_train_smoke_v*/manifest.json')):
        b, tag = load(b_p), b_p.parent.name
        ids = b.get('sample_ids', [])
        check(ids and len(ids) == len(set(ids)) and (not fit or set(ids) <= fit), f'{tag}: sample_ids invalid or outside FIT')
        if split_sha:
            check(b.get('split_manifest_sha256') == split_sha, f'{tag}: split_manifest_sha256 != committed split_v1.json')
        for k in ('catalog_sha256', 'runtime_sha256'):
            check(bool(SHA.match(str(b.get(k)))), f'{tag}: {k} is not sha256')

    # 4) raw data
    raw = [p for p in ROOT.rglob('*') if p.is_file() and
           (p.suffix.lower() in {'.wav', '.tar', '.gz', '.zip'} or p.name == 'state.json')]
    check(not raw, f'Raw data must not be committed: {[str(p.relative_to(REPO)) for p in raw][:8]}')
    if (ROOT / 'synthetic/synthetic_phase1.jsonl').exists():
        warnings.append('data/synthetic/synthetic_phase1.jsonl is the superseded v1 (488); use a data/releases/* release')
    stray = [str(p.relative_to(REPO)) for p in ROOT.rglob('*.jsonl') if 'releases' not in p.parts
             and p.parent.name != 'manifests' and p.name != 'synthetic_phase1.jsonl']
    if stray:
        warnings.append(f'jsonl outside data/releases: {stray[:5]}')

    for w in warnings:
        print('WARNING:', w)
    if errors:
        print('DATA checks FAILED:')
        for e in errors:
            print('  -', e)
        sys.exit(1)
    n_rel = len(list(ROOT.glob('releases/*/release*.json')))
    print(f'DATA checks passed: split FIT {len(fit)} / VAL {len(val)}, {n_rel} release(s) verified, no raw data committed.')
    print('Replay, VAL leakage and task quality need private data (validate_phase1 / verify_bundle / eval.scorer).')


if __name__ == '__main__':
    main()
