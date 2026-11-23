"""Score saved VAL predictions; does not create ToolEnv or load model weights."""
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'ops/src'),str(REPO/'llm/baseline_50_2')]
from lifemate.adapters.baseline50_partitions import load_partition
from lifemate.runtime.configuration import strict_json
from lifemate.runtime.smoke_inputs import digest,manifest
from lifemate.runtime.val_evaluation import evaluate_val


def rows(path):
    return [strict_json(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--private-data',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--diagnostics',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError('Preserve existing evaluation')
    samples,tools=load_partition(args.dataset_dir,args.private_data,'val')
    path=args.private_data/'train_references.json';spec=manifest()['files']['private_data/train_references.json']
    if path.stat().st_size!=spec['bytes'] or digest(path)!=spec['sha256']:
        raise ValueError('References differ from pinned baseline')
    references=strict_json(path.read_text(encoding='utf-8'))['references']
    report=evaluate_val(rows(args.predictions),references,tools,[s.sample_id for s in samples],
        rows(args.diagnostics) if args.diagnostics else None)
    report['provenance']={'predictions_sha256':digest(args.predictions),
        'references_sha256':digest(path),'split_sha256':digest(args.private_data/'split.json'),
        'diagnostics_sha256':digest(args.diagnostics) if args.diagnostics else None,
        'scorer_source_sha256':digest(REPO/'llm/baseline_50_2/baseline50/scoring.py')}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    scores=report.pop('scores')
    for name,value in [('summary.json',report),('scores.json',scores)]:
        (args.output_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('n','complete','score_100','failed_samples')}))
    return 0 if report['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
