"""Pinned partition loader and live VAL/public factories. No gold in model views."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys

from lifemate.adapters.baseline50 import Baseline50Policy, Baseline50System
from lifemate.runtime.contracts import RuntimeConfig, Sample
from lifemate.runtime.smoke_inputs import clean_row, digest, inside, manifest


def load_partition(dataset_dir, private_data, mode):
    if mode not in ('val','public'):
        raise ValueError('Expected val or public')
    from baseline50.io import load_inputs
    private_data=Path(private_data).resolve(); assets=manifest()
    for name in ('split.json','retrieval_index.json'):
        path=private_data/name; expected=assets['files']['private_data/'+name]
        if path.stat().st_size != expected['bytes'] or digest(path)!=expected['sha256']:
            raise ValueError('Private split/index differs from pinned release')
    # Canonical loader pins JSONL/catalog/SDK and returns sanitized inputs only.
    rows,tools=load_inputs({'dataset_dir':Path(dataset_dir).resolve(),
        'assets_dir':private_data.parent},mode,assets)
    root=Path(dataset_dir).resolve()/('train' if mode=='val' else 'public_test')
    samples=[]
    for row in rows:
        clean=clean_row(row)
        for reference in [clean['latest_audio']]+[t['audio'] for t in clean['history'] if t['role']=='user']:
            inside(root,reference)
        samples.append(Sample(clean.pop('id'),clean))
    return samples,tools


@contextmanager
def build_partition_system(config,mode,log_dir):
    from baseline50.io import verify_assets
    from baseline50.asr import MediumASR
    from baseline50.backend import LocalServer
    assets=verify_assets(config)
    samples,tools=load_partition(config['dataset_dir'],config['assets_dir']/'private_data',mode)
    split='train' if mode=='val' else 'public_test'
    if any(n=='zaic_tools' or n.startswith('zaic_tools.') for n in sys.modules):
        raise ValueError('Fresh process required; do not mix SDK splits')
    sdk_root=(config['dataset_dir']/split).resolve();sys.path.insert(0,str(sdk_root))
    sdk=importlib.import_module('zaic_tools')
    if not Path(sdk.__file__).resolve().is_relative_to(sdk_root/'zaic_tools'):
        raise ValueError('Unexpected SDK import origin')
    policy=Baseline50Policy(config['assets_dir']/'private_data')
    policy.identity['assets_verified_by_bridge']=True
    asr=MediumASR(config['assets_dir']/'models/asr',config['dataset_dir'],split,assets['asr_vocab_sha256'])
    with LocalServer(config['assets_dir']/assets['qwen_file'],config['llama_server'],Path(log_dir)) as transport:
        yield Baseline50System(catalog=tools,policy=policy,transport=transport,asr=asr,
            environment_factory=lambda sample:sdk.ToolEnv(split,sample.sample_id,root=config['dataset_dir']),
            allowed_ids=[s.sample_id for s in samples],runtime=RuntimeConfig(),
            provenance={'mode':'baseline50_'+mode,'model_assets_verified':True,'warmup_performed':False,
                        'official_score_claimed':False,'asset_manifest_release':assets['release']})
