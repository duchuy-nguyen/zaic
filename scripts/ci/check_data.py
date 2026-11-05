"""Read-only structural checks of committed synthetic assets; not official scoring."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'data'


def main():
    assets = ROOT / 'synthetic'
    rows = [json.loads(line) for line in (assets / 'synthetic_phase1.jsonl').read_text(encoding='utf-8').splitlines()]
    meta = json.loads((assets / 'synthetic_phase1_meta.json').read_text(encoding='utf-8'))
    catalog = json.loads((assets / 'synthetic_tools_catalog.json').read_text(encoding='utf-8'))
    ids = [r['id'] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('Synthetic IDs must be nonempty and unique')
    if len(rows) != meta['n'] or dict(Counter(r['technique'] for r in rows)) != meta['by_tech']:
        raise ValueError('Synthetic metadata counts do not match records')
    tools = {t['name'] for app in catalog['apps'] for t in app['tools']}
    for row in rows:
        if not isinstance(row['id'], str) or not row['id']:
            raise ValueError('Invalid synthetic ID')
        for call in row['label']['tool_calls']:
            if call['name'] not in tools or not isinstance(call['arguments'], dict):
                raise ValueError('Call is absent from catalog or has invalid arguments shape')
    fit = json.loads((ROOT / 'manifests/train_fit_provisional.json').read_text(encoding='utf-8'))
    if not isinstance(fit, list) or len(fit) != len(set(fit)):
        raise ValueError('Invalid provisional fit manifest')
    print(f'DATA structural checks passed: {len(rows)} synthetic records, {len(tools)} tools')
    print('Official split leakage, runtime replay and task quality require private data and are not checked here.')


if __name__ == '__main__':
    main()
