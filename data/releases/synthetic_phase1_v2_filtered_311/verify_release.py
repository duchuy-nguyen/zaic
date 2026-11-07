"""Verify release integrity; optionally reproduce filtering from the original JSONL."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='Original 331-row JSONL; SHA256 must match release.json')
    parser.add_argument('--output',type=Path,help='New filtered JSONL; requires --source')
    args=parser.parse_args()
    try:
        manifest=json.loads((ROOT/'release.json').read_text(encoding='utf-8'))
        dataset=ROOT/'synthetic_phase1.jsonl'
        exclusions=ROOT/'excluded_samples.csv'
        if sha(dataset)!=manifest['dataset_sha256'] or sha(exclusions)!=manifest['excluded_samples_sha256']:
            raise ValueError('release_hash_mismatch')
        with exclusions.open(encoding='utf-8',newline='') as stream:
            excluded=[r['sample_id'] for r in csv.DictReader(stream)]
        if len(excluded)!=len(set(excluded)) or set(excluded)!=set(manifest['excluded_ids']):
            raise ValueError('exclusion_list_mismatch')
        rows=[json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
        ids=[r['id'] for r in rows]
        if len(rows)!=manifest['records'] or len(ids)!=len(set(ids)) or set(ids)&set(excluded):
            raise ValueError('release_count_or_id_mismatch')
        if dict(Counter(r['technique'] for r in rows))!=manifest['by_technique']:
            raise ValueError('technique_counts_mismatch')
        if args.output and not args.source:
            raise ValueError('output_requires_source')
        if args.source:
            if sha(args.source)!=manifest['source_dataset_sha256']:
                raise ValueError('source_hash_mismatch')
            original=args.source.read_text(encoding='utf-8').splitlines(keepends=True)
            if len(original)!=manifest['source_records']:
                raise ValueError('source_count_mismatch')
            kept=[line for line in original if json.loads(line)['id'] not in set(excluded)]
            reproduced=''.join(kept).encode('utf-8')
            if hashlib.sha256(reproduced).hexdigest()!=manifest['dataset_sha256']:
                raise ValueError('reproduction_hash_mismatch')
            if args.output:
                with args.output.open('xb') as stream:stream.write(reproduced)
        print(json.dumps({'valid':True,'records':len(rows),'excluded':len(excluded),'semantic_quality_certified':False}))
        return 0
    except (OSError,ValueError,TypeError,KeyError):
        print(json.dumps({'valid':False,'error':'invalid_release_or_source_or_output_exists'}))
        return 1

if __name__=='__main__':raise SystemExit(main())
