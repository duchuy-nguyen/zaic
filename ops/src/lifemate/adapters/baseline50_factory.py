"""Opt-in TRAIN-fit smoke factory; validates existing assets, owns server cleanup.

Set LIFEMATE_BASELINE50_CONFIG to the baseline four-path config. This factory
does not download, warm up on extra samples, evaluate VAL or run public_test.
"""
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys

from lifemate.adapters.baseline50 import Baseline50Policy, Baseline50System
from lifemate.runtime.contracts import RuntimeConfig


@contextmanager
def build_system():
    from baseline50.asr import MediumASR
    from baseline50.backend import LocalServer
    from baseline50.io import load_config, read, sha, verify_assets
    config_path = os.environ.get('LIFEMATE_BASELINE50_CONFIG')
    if not config_path:
        raise ValueError('LIFEMATE_BASELINE50_CONFIG required')
    config = load_config(config_path)
    assets = verify_assets(config)
    root = config['dataset_dir'] / 'train'
    spec = assets['dataset']['train']
    if (sha(root / 'train.jsonl') != spec['jsonl_sha256']
            or sha(root / 'train_tools_catalog.json') != spec['catalog_sha256']):
        raise ValueError('TRAIN dataset/catalog differs from frozen baseline')
    runtime = {p.relative_to(root / 'zaic_tools').as_posix(): sha(p)
               for p in sorted((root / 'zaic_tools').rglob('*.py'))}
    if runtime != spec['runtime_py_sha256']:
        raise ValueError('TRAIN SDK differs from frozen baseline')
    if any(n == 'zaic_tools' or n.startswith('zaic_tools.') for n in sys.modules):
        raise ValueError('Fresh process required for pinned TRAIN SDK')
    policy = Baseline50Policy(config['assets_dir'] / 'private_data')
    policy.identity['assets_verified_by_bridge'] = True
    catalog = {t['name']: t for a in read(root / 'train_tools_catalog.json')['apps'] for t in a.get('tools', [])}
    sys.path.insert(0, str(root))
    sdk = importlib.import_module('zaic_tools')
    if not Path(sdk.__file__).resolve().is_relative_to((root / 'zaic_tools').resolve()):
        raise ValueError('Unexpected SDK import origin')
    log_dir = config['output_dir'] / 'ops-train-smoke-server'
    if log_dir.exists():
        raise FileExistsError('Use a new baseline output_dir for smoke server logs')
    asr = MediumASR(config['assets_dir'] / 'models/asr', config['dataset_dir'],
                    'train', assets['asr_vocab_sha256'])
    with LocalServer(config['assets_dir'] / assets['qwen_file'], config['llama_server'], log_dir) as transport:
        yield Baseline50System(catalog=catalog,
            environment_factory=lambda sample: sdk.ToolEnv('train', sample.sample_id, root=config['dataset_dir']),
            transport=transport, policy=policy, asr=asr, runtime=RuntimeConfig(),
            allowed_ids=policy.fit_ids,
            provenance={'mode': 'baseline50_TRAIN_fit_smoke', 'model_assets_verified': True,
                        'assets_manifest_sha256': sha(Path(importlib.import_module('baseline50.io').__file__).parent / 'assets.manifest.json'),
                        'prepared_text_or_audio': 'declared_by_prepared_input_fields',
                        'startup_and_warmup_excluded': True, 'warmup_performed': False,
                        'official_score_claimed': False})
