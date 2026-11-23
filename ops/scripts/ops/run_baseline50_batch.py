"""Explicit VAL/public batch with pinned inputs and fail-closed session resume."""
import argparse
import json
from pathlib import Path
import sys
import uuid

REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'ops/src'),str(REPO/'llm/baseline_50_2')]
from baseline50.io import load_config
from lifemate.adapters.baseline50_partitions import load_partition,build_partition_system
from lifemate.runtime.batch import run_resumable
from lifemate.runtime.smoke_inputs import digest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['val','public'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(argv);config=load_config(args.config)
    samples,_=load_partition(config['dataset_dir'],config['assets_dir']/'private_data',args.mode)
    if args.preflight_only:
        print(json.dumps({'mode':args.mode,'samples':len(samples),'inputs_verified':True,
                          'model_assets_verified':False,'model_inference_run':False}))
        return 0
    source=[p for parent in [REPO/'ops/src',REPO/'ops/scripts',REPO/'llm/baseline_50_2/baseline50']
            for p in parent.rglob('*.py')]
    identity={'mode':args.mode,'config_sha256':digest(args.config),
        'resolved_paths':{k:str(v) for k,v in config.items()},
        'assets_manifest_sha256':digest(REPO/'llm/baseline_50_2/baseline50/assets.manifest.json'),
        'source_sha256':{p.relative_to(REPO).as_posix():digest(p) for p in sorted(source)}}
    # New server log directory each process; journal controls sample continuation.
    factory=lambda:build_partition_system(config,args.mode,args.output_dir/('server-'+uuid.uuid4().hex))
    result=run_resumable(factory,samples,[s.sample_id for s in samples],args.output_dir,identity,resume=args.resume)
    print(json.dumps({k:result[k] for k in ('status','completed','failures')}))
    return 1 if result['failures'] else 0


if __name__=='__main__':raise SystemExit(main())
