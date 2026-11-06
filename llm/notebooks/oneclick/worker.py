"""Stage processes embedded inside LifeMate_OneClick.ipynb.

GPU imports are lazy: core tests and data preparation need no pretrained model.
The notebook launches one process per stage, so CUDA allocations die with it.
"""
from copy import deepcopy
import ast
from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import shutil
import sys
import time
import wave

# -I excludes the script directory and injected Kaggle PYTHONPATH hooks.
# Add only our verified embedded bundle explicitly, after Python startup.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import *
from scoring import score, VERSION as SCORER_VERSION, ASSUMPTIONS
from bounded_attention import completion_loss


class SessionBudget(RuntimeError):
    pass


def budget_guard(reserve_seconds=120):
    value = os.environ.get('LIFEMATE_SESSION_DEADLINE')
    if value and time.time() + reserve_seconds >= float(value):
        raise SessionBudget('SESSION_BUDGET: stopped at a safe boundary; save private resume archive and continue next session')


def deadline_callback():
    from transformers import TrainerCallback
    class DeadlineCallback(TrainerCallback):
        stopped = False
        def on_step_end(self, args, state, control, **kwargs):
            try:
                budget_guard(180)
            except SessionBudget:
                self.stopped = True
                control.should_save = True
                control.should_training_stop = True
            return control
    return DeadlineCallback()


def paths(cfg):
    return Path(cfg['run_dir']), Path(cfg['data_root'])


def load_data(cfg):
    out, root = paths(cfg)
    rows = read_jsonl(root / 'train/train.jsonl')
    catalog = json.loads((root / 'train/train_tools_catalog.json').read_text(encoding='utf-8'))
    tools = {t['name']: t for app in catalog['apps'] for t in app['tools']}
    split = json.loads((out / 'split.json').read_text(encoding='utf-8'))
    return rows, tools, split


def chosen(ids, limit):
    return list(ids) if limit is None else list(ids)[:limit]


def selected_ids(cfg, split, which):
    return chosen(split['fit_ids'] if which == 'fit' else split['val_ids'],
                  cfg['limits']['fit_samples' if which == 'fit' else 'val_samples'])


def runtime(cfg):
    """Load one pinned TRAIN toolkit per process, never the synthetic copy."""
    root = Path(cfg['data_root'])
    if 'zaic_tools' in sys.modules:
        raise RuntimeError('Fresh process required for split-specific ToolEnv')
    sys.path.insert(0, str(root / 'train'))
    from zaic_tools import ToolEnv
    return ToolEnv


def setup_torch(cfg):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(cfg['deterministic'])
    return torch


def logical_parameters(model):
    """Params4bit.numel counts packed storage, not original logical parameters."""
    count = 0
    for p in model.parameters():
        state = getattr(p, 'quant_state', None)
        count += math.prod(state.shape) if state is not None else p.numel()
    return count


def preflight(cfg):
    torch = setup_torch(cfg)
    from transformers import Qwen3_5ForConditionalGeneration, WhisperForConditionalGeneration, Trainer, TrainingArguments, GenerationConfig
    import peft, bitsandbytes
    if torch.cuda.device_count() < cfg['teacher_min_gpus']:
        raise RuntimeError('Teacher requires configured GPU count; choose GPU T4 x2. No automatic model fallback.')
    if 'logits_to_keep' not in inspect.signature(Qwen3_5ForConditionalGeneration.forward).parameters:
        raise RuntimeError('Pinned model class must support selective logits')
    if 'warmup_steps' not in inspect.signature(TrainingArguments).parameters:
        raise RuntimeError('Pinned TrainingArguments must support warmup_steps (ratio as float)')
    if not hasattr(GenerationConfig(), 'prefill_chunk_size'):
        raise RuntimeError('Pinned generation stack must support chunked prefill')
    out, root = paths(cfg)
    data = {'train_sha256': digest(root / 'train/train.jsonl'),
            'catalog_sha256': digest(root / 'train/train_tools_catalog.json'),
            'runtime_sha256': fingerprint({str(p.relative_to(root / 'train/zaic_tools')): digest(p)
                for p in sorted((root / 'train/zaic_tools').rglob('*.py'))})}
    if cfg['expected_train_sha256'] and data['train_sha256'] != cfg['expected_train_sha256']:
        raise ValueError('Unexpected official train manifest revision')
    from huggingface_hub import HfApi, try_to_load_from_cache
    downloads = []
    for role in ('teacher', 'student', 'asr'):
        info = HfApi().model_info(cfg[role + '_id'], revision=cfg[role + '_revision'], files_metadata=True)
        if info.sha != cfg[role + '_revision']:
            raise ValueError('Model revision mismatch')
        weights = [f for f in info.siblings if f.rfilename.endswith('.safetensors') or f.rfilename == 'pytorch_model.bin']
        if not weights or any(f.size is None for f in weights):
            raise ValueError('Cannot establish pinned weight download size')
        cached_bytes = 0
        for file in weights:
            cached = try_to_load_from_cache(info.id, file.rfilename, revision=info.sha)
            if isinstance(cached, str) and Path(cached).is_file() and Path(cached).stat().st_size == file.size:
                cached_bytes += file.size
        downloads.append({'role': role, 'repo': info.id, 'revision': info.sha,
                          'weight_bytes': sum(f.size for f in weights), 'cached_weight_bytes': cached_bytes})
    cache = Path(os.environ['HF_HOME']); cache.mkdir(parents=True, exist_ok=True)
    # Conservative first-run space gate. Cache downloads live outside saved output.
    needed = sum(d['weight_bytes'] - d['cached_weight_bytes'] for d in downloads) * 1.15 + 2 * 2**30
    free = __import__('shutil').disk_usage(cache).free
    if free < needed and not cfg['allow_existing_cache_space_override']:
        raise RuntimeError('Insufficient scratch for all pinned weights + margin. Free space/use prepared cache; no silent deletion.')
    report = {'data': data, 'config_sha256': fingerprint(cfg), 'code_version': VERSION,
        'downloads': downloads, 'scratch_free_gib': free/2**30, 'remaining_download_required_gib': needed/2**30,
        'python': sys.version, 'packages': {p: version(p) for p in
            ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'huggingface-hub', 'tokenizers', 'safetensors')},
        'gpus': [{'name': torch.cuda.get_device_name(i), 'capability': list(torch.cuda.get_device_capability(i)),
            'free_gib': torch.cuda.mem_get_info(i)[0] / 2**30, 'total_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
            for i in range(torch.cuda.device_count())], 'cuda': torch.version.cuda,
        'quality_claim': 'No GPU/model quality measurement yet; pilot is not full validation',
        'scorer': SCORER_VERSION, 'scorer_assumptions': ASSUMPTIONS}
    previous = out / 'provenance.json'
    if previous.exists():
        old = json.loads(previous.read_text(encoding='utf-8'))
        for key in ('data', 'packages', 'cuda', 'config_sha256', 'code_version'):
            if old[key] != report[key]:
                raise RuntimeError('Resume provenance changed: ' + key + '; use a separate run')
        if [(g['name'], g['capability']) for g in old['gpus']] != [(g['name'], g['capability']) for g in report['gpus']]:
            raise RuntimeError('GPU type changed on resume; preserve old run and start a new one')
    write_json(out / 'provenance.json', report)
    print(json.dumps({'preflight': 'pass', 'gpus': report['gpus'], 'mode': cfg['mode']}), flush=True)


def prepare(cfg):
    out, root = paths(cfg)
    rows = read_jsonl(root / 'train/train.jsonl')
    split = make_split(rows, root, cfg['seed'], cfg['val_fraction'])
    write_json(out / 'split.json', split)
    fit = utterances(rows, selected_ids(cfg, split, 'fit'), split)
    val = utterances(rows, selected_ids(cfg, split, 'val'), split)
    if {r['audio_sha256'] for r in fit} & {r['audio_sha256'] for r in val}:
        raise ValueError('ASR audio leakage between partitions')
    write_json(out / 'asr_fit.json', fit)
    write_json(out / 'asr_val.json', val)
    report = {'total_train_samples': len(rows), 'fit_samples': len(split['fit_ids']),
        'val_samples': len(split['val_ids']), 'selected_fit_samples': len(selected_ids(cfg, split, 'fit')),
        'selected_val_samples': len(selected_ids(cfg, split, 'val')), 'asr_fit_utterances': len(fit),
        'asr_val_utterances': len(val), 'split_sha256': digest(out / 'split.json'),
        'largest_group': split['largest_group'], 'limitations': split['limitations']}
    write_json(out / 'data_report.json', report)
    print(json.dumps(report), flush=True)


def reuse_asr(cfg):
    """Import completed, compatible ASR artifacts with explicit source provenance."""
    out, _ = paths(cfg)
    source_name = cfg.get('reuse_asr_from')
    if not source_name:
        write_json(out / 'asr_reuse.json', {'enabled': False})
        return
    source = Path(source_name).resolve()
    if source == out.resolve(): raise ValueError('ASR reuse requires a different source run')
    previous = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    keys = ('mode', 'seed', 'deterministic', 'val_fraction', 'data_root', 'expected_train_sha256',
        'asr_id', 'asr_revision', 'asr_lora_r', 'asr_grad_accum', 'asr_lr', 'asr_epochs',
        'asr_max_new_tokens', 'asr_eval_seconds', 'save_steps')
    for key in keys:
        if previous[key] != cfg[key]: raise ValueError('Incompatible ASR reuse config: ' + key)
    for key in ('fit_samples', 'val_samples', 'asr_max_steps'):
        if previous['limits'][key] != cfg['limits'][key]: raise ValueError('Incompatible ASR reuse limits: ' + key)
    before = json.loads((source / 'provenance.json').read_text(encoding='utf-8'))
    current = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
    for key in ('data', 'packages', 'cuda'):
        if before[key] != current[key]: raise ValueError('Incompatible ASR provenance: ' + key)
    if digest(source / 'split.json') != digest(out / 'split.json'):
        raise ValueError('ASR reuse split differs; no artifacts imported')
    members=previous.get('bundle_files', ('core.py', 'worker.py', 'scoring.py'))
    if set(members)-{'core.py','worker.py','scoring.py','bounded_attention.py','verify_training_stack.py'}:
        raise ValueError('Unexpected ASR source bundle member')
    bundle = {name: (source / 'source' / name).read_text(encoding='utf-8') for name in members}
    raw = json.dumps(bundle, ensure_ascii=False, sort_keys=True).encode()
    if hashlib.sha256(raw).hexdigest() != previous['bundle_sha256']:
        raise ValueError('ASR source snapshot does not match source config bundle hash')
    if bundle['core.py'] != Path(__file__).with_name('core.py').read_text(encoding='utf-8'):
        raise ValueError('ASR shared core source changed; start a measured ASR run instead')
    def functions(text):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    old_functions = functions(bundle['worker.py'])
    new_functions = functions(Path(__file__).read_text(encoding='utf-8'))
    for name in ('setup_torch', 'paths', 'read_wave', 'load_asr', 'asr_eval', 'transcribe', 'stopping',
                 'trainer_args', 'last_checkpoint', 'asr_train', 'logical_parameters', 'deadline_callback', 'budget_guard'):
        if old_functions.get(name) != new_functions.get(name):
            raise ValueError('ASR source function changed: ' + name)
    verified = {}
    source_identity = fingerprint(previous)
    markers = {}
    for stage in ('asr_base', 'asr_train', 'asr_tuned'):
        marker = json.loads((source / ('stage_' + stage + '.json')).read_text(encoding='utf-8'))
        if marker['status'] != 'completed' or marker['identity'] != source_identity:
            raise ValueError('ASR source stage incomplete/incompatible: ' + stage)
        for name, sha in marker['artifacts'].items():
            path = (source / name).resolve()
            if not path.is_relative_to(source) or not (out / name).resolve().is_relative_to(out.resolve()):
                raise ValueError('Unsafe ASR artifact path')
            if not (name.startswith('asr_adapter/') or name.startswith('asr_') or name == 'selected_asr.json'):
                raise ValueError('Unexpected ASR artifact: ' + name)
            if digest(path) != sha: raise ValueError('ASR artifact changed: ' + name)
            if (out / name).exists() and digest(out / name) != sha:
                raise ValueError('Refusing overwrite of differing ASR artifact: ' + name)
            verified[name] = sha
        markers[stage] = marker
    if 'asr_adapter/adapter_model.safetensors' not in verified:
        raise ValueError('Verified ASR adapter weights missing')
    # All compatibility and hash checks complete before copying or writing markers.
    for name in verified:
        target = out / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    receipt = {'enabled': True, 'source_run': str(source), 'source_identity': source_identity,
        'source_bundle_sha256': previous['bundle_sha256'], 'artifacts': verified,
        'source_provenance': before,
        'note': 'Imported completed ASR; not retrained. Teacher/student artifacts not imported.'}
    write_json(out / 'asr_reuse.json', receipt)
    for stage, marker in markers.items():
        write_json(out / ('stage_' + stage + '.json'), {'identity': fingerprint(cfg),
            'stage': stage, 'status': 'completed', 'origin': 'verified_asr_import',
            'source_identity': source_identity, 'source_run': str(source), 'artifacts': marker['artifacts']})
    print('Reused verified completed ASR:', source, '| no ASR retraining', flush=True)


def reuse_teacher(cfg):
    """Reuse immutable completed development trajectories; never replay a sample."""
    out, _ = paths(cfg)
    if not cfg.get('reuse_teacher_from'):
        write_json(out / 'teacher_reuse.json', {'enabled': False})
        return
    source = Path(cfg['reuse_teacher_from']).resolve()
    previous = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    for key in ('teacher_id', 'teacher_revision', 'teacher_prequantized', 'seed', 'data_root'):
        if previous[key] != cfg[key]: raise ValueError('Teacher reuse config mismatch: ' + key)
    for name in ('split.json', 'references.jsonl'):
        if digest(source / name) != digest(out / name): raise ValueError('Teacher reuse dataset differs: ' + name)
    a = json.loads((source / 'provenance.json').read_text(encoding='utf-8'))
    b = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
    if a['data'] != b['data']: raise ValueError('Teacher toolkit/data provenance changed')
    files = previous.get('bundle_files', ('core.py', 'worker.py', 'scoring.py'))
    if set(files) - {'core.py', 'worker.py', 'scoring.py', 'bounded_attention.py', 'verify_training_stack.py'}:
        raise ValueError('Unexpected teacher source bundle member')
    bundle = {name: (source / 'source' / name).read_text(encoding='utf-8') for name in files}
    if hashlib.sha256(json.dumps(bundle, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != previous['bundle_sha256']:
        raise ValueError('Teacher source snapshot mismatch')
    marker = json.loads((source / 'stage_teacher.json').read_text(encoding='utf-8'))
    if marker['status'] != 'completed' or marker['identity'] != fingerprint(previous):
        raise ValueError('Teacher source stage must be completed')
    allowed = {'teacher_rollouts.jsonl', 'teacher_metrics.json', 'teacher_memory_smoke.json'}
    for name, sha in marker['artifacts'].items():
        if name not in allowed or digest(source / name) != sha:
            raise ValueError('Teacher artifact missing/changed: ' + name)
        if (out / name).exists() and digest(out / name) != sha:
            raise ValueError('Refusing to overwrite teacher artifact: ' + name)
    rows, catalog, split = load_data(cfg)
    ids = chosen(selected_ids(cfg, split, 'fit'), cfg['limits']['teacher_samples'])
    records = read_jsonl(source / 'teacher_rollouts.jsonl')
    if [r['id'] for r in records] != ids: raise ValueError('Teacher IDs differ from frozen fit selection')
    refs = {r['id']: r for r in read_jsonl(out / 'references.jsonl')}
    accepted = 0
    for record in records:
        value = score(record, refs[record['id']], catalog, failed=failed_status(record['status']))
        valid = bool(value['state_match'] and value['score_spec_local'] >= cfg['teacher_min_score']
                     and record['result_alignment_known'] and not failed_status(record['status']))
        if valid != record['accepted']: raise ValueError('Teacher revalidation changed acceptance')
        accepted += valid
    if accepted < cfg['minimum_teacher_accepted']: raise ValueError('Too few accepted source teacher trajectories')
    for name in marker['artifacts']: shutil.copy2(source / name, out / name)
    write_json(out / 'teacher_reuse.json', {'enabled': True, 'source_run': str(source),
        'teacher_id': previous['teacher_id'], 'teacher_revision': previous['teacher_revision'],
        'source_identity': fingerprint(previous), 'accepted': accepted,
        'source_bundle_sha256': previous['bundle_sha256'],
        'note': 'Historical validated training data, not a benchmark of the new controller/attention backend.'})
    write_json(out / 'stage_teacher.json', {'status': 'completed', 'stage': 'teacher',
        'identity': fingerprint(cfg), 'origin': 'verified_teacher_data_import',
        'source_identity': fingerprint(previous), 'artifacts': marker['artifacts']})
    print('Reused verified teacher trajectories:', accepted, '/', len(records), '| no teacher generation', flush=True)


def references(cfg):
    out, root = paths(cfg)
    rows, tools, split = load_data(cfg)
    wanted = set(selected_ids(cfg, split, 'fit') + selected_ids(cfg, split, 'val'))
    factory = runtime(cfg)
    from zaic_tools import canon
    target = out / 'references.jsonl'
    saved = read_jsonl(target) if target.exists() else []
    remaining = [r for r in rows if r['id'] in wanted and r['id'] not in {x['id'] for x in saved}]
    for row in remaining:
        budget_guard(30)
        env = factory('train', row['id'], root=root)  # independent gold annotation job
        for call in row['label']['tool_calls']:
            if not env.validate(call['name'], call['arguments'])['ok']:
                raise ValueError('Gold call invalid for pinned toolkit')
            result = env.call(call['name'], call['arguments'])
            if not result.get('ok'):
                raise ValueError('Gold replay rejected')
        partial = row['label']['final_state']
        # Explicit train-only reference check. Never exposed to an agent/model.
        final = env.get_state()
        consistent = canon(partial) == canon({app: final[app] for app in partial})
        record = {'id': row['id'], 'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash(),
                  'partial_label_consistent': consistent}
        append_jsonl(target, record)
        if not consistent and cfg['strict_reference_consistency']:
            raise ValueError('Gold replay/partial final_state mismatch; inspect private references artifact')
    records = read_jsonl(target)
    if cfg['strict_reference_consistency'] and any(not r['partial_label_consistent'] for r in records):
        raise ValueError('Existing reference mismatch; do not silently train on inconsistent labels')
    print('TRAIN references prepared:', len(records), flush=True)


def read_wave(root, relative):
    import numpy as np
    path = audio_path(root, relative)
    wav_meta(path)
    with wave.open(str(path), 'rb') as stream:
        expected = stream.getnframes()
        data = stream.readframes(expected)
    if len(data) != expected * 2:
        raise ValueError('Truncated PCM waveform')
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768


def load_asr(cfg, train=False, adapter=None):
    torch = setup_torch(cfg)
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model, PeftModel
    processor = WhisperProcessor.from_pretrained(cfg['asr_id'], revision=cfg['asr_revision'])
    processor.tokenizer.set_prefix_tokens(language='vi', task='transcribe', predict_timestamps=False)
    dtype = torch.float32 if train else torch.float16
    model = WhisperForConditionalGeneration.from_pretrained(cfg['asr_id'], revision=cfg['asr_revision'],
        dtype=dtype, attn_implementation='sdpa').to('cuda:0')
    model.generation_config.language = 'vi'; model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = None
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    elif train:
        # Generic PeftModel avoids Seq2Seq wrapper injecting input_ids into Whisper.
        model = get_peft_model(model, LoraConfig(r=cfg['asr_lora_r'], lora_alpha=cfg['asr_lora_r']*2,
            lora_dropout=0., bias='none', target_modules=['q_proj', 'v_proj']))
        model.config.use_cache = False
    return model, processor


def stopping(deadline):
    from transformers import StoppingCriteria, StoppingCriteriaList
    class WallClock(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return time.monotonic() >= deadline
    return StoppingCriteriaList([WallClock()])


def transcribe(model, processor, cfg, relative, deadline):
    import torch
    if time.monotonic() >= deadline:
        raise TimeoutError('ASR deadline exhausted')
    waveform = read_wave(Path(cfg['data_root']), relative)
    batch = processor.feature_extractor(waveform, sampling_rate=16000, return_tensors='pt',
        padding='max_length', truncation=False, return_attention_mask=True)
    features = batch.input_features.to('cuda:0', dtype=torch.float16)
    mask = batch.attention_mask.to('cuda:0')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        generated = model.generate(input_features=features, attention_mask=mask, language='vi', task='transcribe',
            do_sample=False, num_beams=1, max_new_tokens=cfg['asr_max_new_tokens'], return_dict_in_generate=True,
            return_timestamps=False, stopping_criteria=stopping(deadline))
    torch.cuda.synchronize()
    if time.monotonic() >= deadline:
        raise TimeoutError('ASR response after deadline')
    tokens = generated.sequences[0].tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    if not eos.intersection(tokens):
        raise ValueError('ASR output truncated without EOS')
    return norm(processor.tokenizer.decode(tokens, skip_special_tokens=True))


def asr_eval(cfg, trained=False):
    out, _ = paths(cfg)
    model, processor = load_asr(cfg, adapter=out / 'asr_adapter' if trained else None)
    model.eval()
    rows = json.loads((out / 'asr_val.json').read_text(encoding='utf-8'))
    target = out / ('asr_tuned_eval.jsonl' if trained else 'asr_base_eval.jsonl')
    records = read_jsonl(target) if target.exists() else []
    seen = {r['audio_sha256'] for r in records}
    if len(seen) != len(records) or seen - {r['audio_sha256'] for r in rows}:
        raise ValueError('Unexpected/duplicate ASR resume identities')
    for row in rows:
        if row['audio_sha256'] in seen:
            continue
        budget_guard(cfg['asr_eval_seconds'] + 120)
        started, text, error = time.monotonic(), '', None
        try:
            text = transcribe(model, processor, cfg, row['audio'], started + cfg['asr_eval_seconds'])
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        record = {'audio_sha256': row['audio_sha256'], 'reference': row['text'], 'hypothesis': text,
                  'error': error, 'elapsed_s': time.monotonic()-started}
        records.append(record); append_jsonl(target, record)
    metrics = asr_metrics(records)
    metrics['logical_parameters'] = logical_parameters(model)
    metrics['mode'] = cfg['mode']; metrics['asr_id'] = cfg['asr_id']
    write_json(out / ('asr_tuned_metrics.json' if trained else 'asr_base_metrics.json'), metrics)
    if trained:
        base = json.loads((out / 'asr_base_metrics.json').read_text(encoding='utf-8'))
        if base['reference_words'] != metrics['reference_words'] or base['utterances'] != metrics['utterances']:
            raise ValueError('ASR validation denominators changed')
        use_tuned = metrics['errors'] <= base['errors'] and (metrics['wer'], metrics['cer']) < (base['wer'], base['cer'])
        write_json(out / 'selected_asr.json', {'adapter': 'asr_adapter' if use_tuned else None,
            'logical_parameters': metrics['logical_parameters'] if use_tuned else base['logical_parameters'],
            'criterion': 'same validation utterances; error count non-increasing, lexicographic WER/CER improvement',
            'base': base, 'tuned': metrics, 'limitations': 'ASR selection; downstream action gains checked separately'})
    print('ASR evaluation:', json.dumps(metrics), flush=True)


def trainer_args(cfg, directory, asr=False):
    from transformers import TrainingArguments
    return TrainingArguments(output_dir=str(directory), per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg['asr_grad_accum' if asr else 'student_grad_accum'],
        learning_rate=cfg['asr_lr' if asr else 'student_lr'],
        num_train_epochs=cfg['asr_epochs' if asr else 'student_epochs'],
        max_steps=cfg['limits']['asr_max_steps' if asr else 'student_max_steps'],
        warmup_steps=.05, lr_scheduler_type='cosine', optim='adamw_torch', fp16=True, bf16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        logging_steps=1 if cfg['mode'] == 'pilot' else 10, save_strategy='steps',
        save_steps=cfg['save_steps'], save_total_limit=2, eval_strategy='no', report_to=[],
        remove_unused_columns=False, seed=cfg['seed'], data_seed=cfg['seed'],
        dataloader_num_workers=0, dataloader_pin_memory=False)


def last_checkpoint(directory):
    candidates = sorted(Path(directory).glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    for path in reversed(candidates):
        if (path / 'trainer_state.json').is_file() and (path / 'optimizer.pt').is_file() and (path / 'adapter_model.safetensors').is_file():
            return str(path)
    return None


def asr_train(cfg):
    import torch
    from transformers import Trainer
    out, root = paths(cfg)
    model, processor = load_asr(cfg, train=True)
    rows = json.loads((out / 'asr_fit.json').read_text(encoding='utf-8'))
    class AudioDataset:
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            features = processor.feature_extractor(read_wave(root, row['audio']), sampling_rate=16000,
                return_tensors='np', padding='max_length', truncation=False).input_features[0]
            labels = processor.tokenizer(row['text']).input_ids
            if len(labels) > model.config.max_target_positions:
                raise ValueError('ASR label exceeds decoder budget; do not truncate')
            return {'input_features': features, 'labels': labels}
    def collate(batch):
        labels = processor.tokenizer.pad({'input_ids': [x['labels'] for x in batch]}, return_tensors='pt')
        target = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        if (target[:, 0] == model.config.decoder_start_token_id).all(): target = target[:, 1:]
        return {'input_features': torch.tensor(__import__('numpy').stack([x['input_features'] for x in batch])), 'labels': target}
    directory = out / 'asr_checkpoints'
    trainer = Trainer(model=model, args=trainer_args(cfg, directory, asr=True),
                      train_dataset=AudioDataset(), data_collator=collate)
    trainer.model_accepts_loss_kwargs = False
    callback = deadline_callback(); trainer.add_callback(callback)
    trainer.train(resume_from_checkpoint=last_checkpoint(directory))
    if callback.stopped:
        raise SessionBudget('SESSION_BUDGET: ASR optimizer checkpoint saved; resume training next session')
    trainer.save_model(str(out / 'asr_adapter')); processor.save_pretrained(out / 'asr_adapter')
    trainer.save_state()
    write_json(out / 'asr_training.json', {'steps': trainer.state.global_step,
        'train_utterances': len(rows), 'logical_parameters': logical_parameters(model),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})


def load_llm(cfg, teacher=False, train=False, adapter=None):
    torch = setup_torch(cfg)
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, PeftModel
    model_id, revision = (cfg['teacher_id'], cfg['teacher_revision']) if teacher else (cfg['student_id'], cfg['student_revision'])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    from bounded_attention import register
    backend = register(cfg.get('attention_query_chunk', 128))
    kwargs = {'revision': revision, 'dtype': torch.float16,
              'attn_implementation': backend, 'trust_remote_code': False}
    if not (teacher and cfg['teacher_prequantized']):
        kwargs['quantization_config'] = quant
    if teacher:
        fraction = cfg.get('teacher_gpu_memory_fraction', .85)
        if not 0.5 <= fraction <= .90:
            raise ValueError('Teacher weight-memory fraction must be between 0.5 and 0.90')
        kwargs.update(device_map='balanced', max_memory={i: int(torch.cuda.mem_get_info(i)[0] * fraction)
                                                    for i in range(torch.cuda.device_count())})
    else:
        kwargs['device_map'] = {'': 0}  # training is deliberately single-GPU, not auto/DDP
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if teacher and any(str(v) in ('cpu', 'disk') for v in getattr(model, 'hf_device_map', {}).values()):
        raise RuntimeError('Teacher spilled to CPU/disk; no silent offload performance change')
    if train:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False})
        if cfg.get('student_compact_frozen_io',True):
            # PEFT upcasts frozen embedding/head to FP32. These weights were already
            # loaded as FP16; restore their storage dtype without touching adapters
            # or normalization weights. Save gigabytes for Qwen's 248K vocabulary.
            saved=0;seen=set()
            for module in (model.get_input_embeddings(),model.get_output_embeddings()):
                if module is None: continue
                for p in module.parameters():
                    if id(p) in seen: continue
                    seen.add(id(p))
                    if p.requires_grad: raise ValueError('Frozen IO compaction requires untrained embedding/head')
                    if p.dtype==torch.float32 and getattr(p,'quant_state',None) is None:
                        saved+=p.numel()*2;p.data=p.data.to(torch.float16)
            print('Student frozen embedding/head restored to FP16; saved GiB:',round(saved/2**30,3),flush=True)
        endings = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
                   'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj'}
        targets = [name for name, module in model.named_modules() if '.language_model.' in name
                   and name.rsplit('.', 1)[-1] in endings and hasattr(module, 'weight')]
        if not targets:
            raise ValueError('No language-only LoRA target modules matched pinned architecture')
        model = get_peft_model(model, LoraConfig(r=cfg['student_lora_r'], lora_alpha=cfg['student_lora_r']*2,
            lora_dropout=0., bias='none', target_modules=targets))
        model.config.use_cache = False
        model.config.text_config.use_cache = False
    elif adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    original_eos = model.generation_config.eos_token_id
    model.generation_config.eos_token_id = generation_end_ids(model, tokenizer)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    print('LLM chat termination:', json.dumps({'checkpoint_eos': original_eos,
        'tokenizer_eos': tokenizer.eos_token_id,
        'generation_eos': model.generation_config.eos_token_id}), flush=True)
    return model, tokenizer


def generation_end_ids(model, tokenizer):
    """Checkpoint EOS may be endoftext while chat/tokenizer EOS is im_end."""
    eos = model.generation_config.eos_token_id
    values = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    values.append(tokenizer.eos_token_id)
    ids = sorted({value for value in values if type(value) is int and value >= 0})
    if not ids: raise ValueError('No usable LLM generation end token')
    return ids


def llm_inputs(model, tokenizer, cfg, messages, tools, teacher=False):
    max_input = cfg['teacher_context'] if teacher else cfg['inference_context']
    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
        add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
    ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    if ids.input_ids.shape[1] + cfg['max_new_tokens'] > max_input:
        raise ValueError('Context overflow; no silent truncation')
    device = model.get_input_embeddings().weight.device
    return {k: v.to(device) for k, v in ids.items()}


def generation_memory_options(cfg, teacher=False):
    options = {'logits_to_keep': 1}
    if teacher:
        chunk = cfg.get('teacher_prefill_chunk_size', 256)
        if type(chunk) is not int or chunk < 1:
            raise ValueError('Teacher prefill chunk size must be a positive integer')
        options['prefill_chunk_size'] = chunk
    return options


def gpu_memory_snapshot():
    import torch
    return [{'gpu': i, 'allocated_gib': torch.cuda.memory_allocated(i)/2**30,
             'reserved_gib': torch.cuda.memory_reserved(i)/2**30,
             'peak_allocated_gib': torch.cuda.max_memory_allocated(i)/2**30,
             'free_gib': torch.cuda.mem_get_info(i)[0]/2**30}
            for i in range(torch.cuda.device_count())]


def teacher_memory_smoke(model, tokenizer, cfg, rows, catalog):
    """One-token generation on longest initial fit prompt, without creating ToolEnv."""
    import torch
    out, _ = paths(cfg)
    # Compare lengths on CPU so no full prompt tensors accumulate on the GPU.
    candidates = []
    for row in rows:
        tools = schemas(row, catalog)
        prompt = tokenizer.apply_chat_template(initial_messages(row), tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
        candidates.append((len(tokenizer.encode(prompt, add_special_tokens=False)), row))
    if not candidates: raise ValueError('No fit samples for teacher memory smoke')
    length, row = max(candidates, key=lambda pair: pair[0])
    started = time.monotonic()
    before = gpu_memory_snapshot()
    for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)
    report = {'id': row['id'], 'initial_prompt_tokens': length, 'before': before,
              'device_map': {k: str(v) for k, v in getattr(model, 'hf_device_map', {}).items()},
              'generation_options': generation_memory_options(cfg, teacher=True),
              'limitation': 'Longest initial selected prompt only; later tool observations can increase memory.'}
    try:
        ids = llm_inputs(model, tokenizer, cfg, initial_messages(row), schemas(row, catalog), teacher=True)
        with torch.inference_mode():
            output = model.generate(**ids, do_sample=False, max_new_tokens=1, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=generation_end_ids(model, tokenizer),
                **generation_memory_options(cfg, teacher=True))
        for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)
        del output, ids
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        report.update(after=gpu_memory_snapshot(), elapsed_s=time.monotonic()-started)
        write_json(out / 'teacher_memory_smoke.json', report)
        print('Teacher memory smoke:', json.dumps(report, ensure_ascii=False), flush=True)


def generator(model, tokenizer, cfg, teacher=False):
    import torch
    def generate(messages, tools, deadline):
        ids = llm_inputs(model, tokenizer, cfg, messages, tools, teacher=teacher)
        eos_ids = generation_end_ids(model, tokenizer)
        if time.monotonic() >= deadline:
            raise TimeoutError('Generation deadline exhausted')
        with torch.inference_mode():
            result = model.generate(**ids, do_sample=False, max_new_tokens=cfg['max_new_tokens'],
                use_cache=True, pad_token_id=tokenizer.pad_token_id, stopping_criteria=stopping(deadline),
                eos_token_id=eos_ids,
                **generation_memory_options(cfg, teacher=teacher))
        tokens = result[0, ids['input_ids'].shape[1]:].tolist()
        if time.monotonic() >= deadline:
            raise TimeoutError('Generation finished after deadline')
        if not tokens or tokens[-1] not in eos_ids:
            raise ValueError('Generation hit token cap/no EOS; output rejected')
        while tokens and tokens[-1] in eos_ids: tokens.pop()
        # Retain native tool/control tags; skip_special_tokens can erase them.
        return tokenizer.decode(tokens, skip_special_tokens=False)
    return generate


def guarded_records(out, tag, ids):
    target, active = out / (tag + '.jsonl'), out / (tag + '.inflight.json')
    records = read_jsonl(target) if target.exists() else []
    if [r['id'] for r in records] != ids[:len(records)]:
        raise ValueError('Resume must be the exact frozen ID prefix')
    if active.exists():
        sid = json.loads(active.read_text(encoding='utf-8'))['id']
        if sid not in {r['id'] for r in records}:
            raise RuntimeError('Uncommitted ToolEnv session: ' + sid + '. Refusing reset/rerun; preserve evidence and review.')
        active.unlink()
    return target, active, records


def teacher_diagnostics(records, threshold):
    """Read existing committed rollout metadata; never rerun or reset ToolEnv."""
    reasons = Counter()
    for record in records:
        if record['accepted']:
            continue
        value = record['local_score']
        if not value['state_match']: reasons['state_mismatch'] += 1
        if value['score_spec_local'] < threshold: reasons['score_below_threshold'] += 1
        if not record['result_alignment_known']: reasons['result_alignment_unknown'] += 1
        if failed_status(record['status']): reasons['failed_status'] += 1
    scores = [r['local_score']['score_spec_local'] for r in records]
    elapsed = [r['elapsed_s'] for r in records]
    return {'statuses': dict(Counter(r['status'] for r in records)),
        'rejection_reasons_nonexclusive': dict(reasons),
        'errors': dict(Counter(r['error'] for r in records if r.get('error'))),
        'repair_errors': dict(Counter(t['repair_error'] for r in records
            for t in r.get('trace', []) if 'repair_error' in t)),
        'state_matches': sum(r['local_score']['state_match'] for r in records),
        'score_max': max(scores, default=0), 'score_mean': statistics.mean(scores) if scores else 0,
        'latency_median_s': statistics.median(elapsed) if elapsed else 0,
        'latency_max_s': max(elapsed, default=0),
        'note': 'Counts explain the gate; no threshold/model/time-budget changes are inferred.'}


def teacher(cfg):
    out, root = paths(cfg)
    rows, catalog, split = load_data(cfg)
    ids = chosen(selected_ids(cfg, split, 'fit'), cfg['limits']['teacher_samples'])
    target, active, saved = guarded_records(out, 'teacher_rollouts', ids)
    refs = {r['id']: r for r in read_jsonl(out / 'references.jsonl')}
    # A failed quality gate with all records present needs only a report on rerun.
    # Do not reload 27B weights or replay the already committed samples.
    if len(saved) < len(ids):
        model, tokenizer = load_llm(cfg, teacher=True)
        model.eval()
        teacher_memory_smoke(model, tokenizer, cfg, [r for r in rows if r['id'] in set(ids)], catalog)
        generate = generator(model, tokenizer, cfg, teacher=True)
        factory = runtime(cfg)
    by_id = {r['id']: r for r in rows}
    for sid in ids[len(saved):]:
        budget_guard(cfg['teacher_seconds'] + 120)
        write_json(active, {'id': sid, 'purpose': 'teacher train-fit rollout'})
        env = factory('train', sid, root=root)
        result = run_agent(by_id[sid], catalog, env, generate, seconds=cfg['teacher_seconds'],
                           max_steps=cfg['max_steps'], max_calls=cfg['max_calls'])
        value = score(result, refs[sid], catalog, failed=failed_status(result['status']))
        result['local_score'] = value
        result['accepted'] = (value['state_match'] and value['score_spec_local'] >= cfg['teacher_min_score']
                              and result['result_alignment_known'] and not failed_status(result['status']))
        append_jsonl(target, result); active.unlink()
        print('Teacher:', len(saved)+1, '/', len(ids), 'accepted=', result['accepted'],
              'status=', result['status'], 'score=', value['score_spec_local'],
              'state_match=', value['state_match'], 'seconds=', round(result['elapsed_s'], 2),
              'error=', result.get('error'), flush=True)
        saved.append(result)
        if result['status'] == 'model_error' and 'OutOfMemoryError' in (result.get('error') or ''):
            # Persist this real attempt and stop, rather than consuming the entire fit
            # subset on the same resource failure. No retry/reset of this environment.
            print('Teacher OOM: stop after committed record; preserve log/hash and review memory smoke.', flush=True)
            break
    accepted = sum(r['accepted'] for r in saved)
    write_json(out / 'teacher_metrics.json', {'tasks': len(saved), 'accepted': accepted,
        'acceptance_rate': accepted / max(1, len(saved)), 'scorer': SCORER_VERSION,
        'threshold': cfg['teacher_min_score'], 'not_official_score': True,
        'seconds_total': sum(r['elapsed_s'] for r in saved),
        'diagnostics': teacher_diagnostics(saved, cfg['teacher_min_score'])})
    print('Teacher diagnostics:', json.dumps(teacher_diagnostics(saved, cfg['teacher_min_score']), ensure_ascii=False), flush=True)
    if len(saved) < len(ids):
        raise RuntimeError('Teacher rollout stopped on CUDA OOM; committed records preserved. '
            'Inspect teacher_memory_smoke.json and full error; do not retry/reset samples automatically.')
    if accepted < cfg['minimum_teacher_accepted']:
        raise RuntimeError('Insufficient verified teacher data: ' + str(accepted) + '/' + str(len(saved))
            + ' accepted, required=' + str(cfg['minimum_teacher_accepted'])
            + '. Inspect teacher_metrics.json diagnostics and committed teacher_rollouts.jsonl; '
            + 'do not silently call gold-only training distillation')


def make_examples(cfg):
    out, _ = paths(cfg)
    rows, catalog, split = load_data(cfg)
    fit = set(selected_ids(cfg, split, 'fit')); val = set(split['val_ids'])
    by_id = {r['id']: r for r in rows}
    records = []
    for sid in sorted(fit):
        row = by_id[sid]
        records += supervised_examples(row, schemas(row, catalog), row['label']['tool_calls'], row['label']['tool_results'], 'gold')
    teachers = read_jsonl(out / 'teacher_rollouts.jsonl')
    for result in teachers:
        if result['id'] not in fit or result['id'] in val:
            raise ValueError('Teacher source outside train-fit')
        if result['accepted']:
            row = by_id[result['id']]
            records += supervised_examples(row, schemas(row, catalog), result['tool_calls'], result['results'], 'teacher')
    # Identical correct teacher outputs add no new hard-target information.
    # Keep both provenance labels but train the exact same example only once.
    dedup = {}
    for record in records:
        key = fingerprint({k: record[k] for k in ('source_id', 'messages', 'tools', 'target')})
        if key in dedup:
            dedup[key]['sources'].append(record['source'])
        else:
            dedup[key] = {**record, 'sources': [record['source']]}
    target = out / 'student_examples.jsonl'
    with target.open('w', encoding='utf-8') as stream:
        for record in dedup.values():
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    report = {'raw_examples': len(records), 'unique_examples': len(dedup),
        'raw_by_source': dict(Counter(r['source'] for r in records)),
        'unique_with_teacher_provenance': sum('teacher' in r['sources'] for r in dedup.values()),
        'teacher_only_unique_examples': sum(r['sources'] == ['teacher'] for r in dedup.values()),
        'note': 'Identical teacher/gold hard targets are deduplicated; no claim of extra knowledge from copies.'}
    write_json(out / 'examples_report.json', report)
    print(json.dumps(report), flush=True)


def tokenized_training(cfg, tokenizer):
    out, _ = paths(cfg)
    records = read_jsonl(out / 'student_examples.jsonl')
    encoded, rejected, lengths = [], [], []
    for record in records:
        tokens, length = training_tokens(tokenizer, record, cfg['student_train_context'])
        lengths.append(length)
        if tokens is None:
            rejected.append({'source_id': record['source_id'], 'tokens': length, 'sources': record['sources']})
        else:
            encoded.append(tokens)
    coverage = len(encoded) / max(1, len(records))
    write_json(out / 'length_report.json', {'examples': len(records), 'accepted': len(encoded),
        'coverage': coverage, 'context': cfg['student_train_context'], 'max_tokens': max(lengths, default=0),
        'rejected': rejected, 'no_truncation': True, 'selection_bias_warning': 'Long examples are excluded explicitly; inspect source coverage.'})
    if not encoded or coverage < cfg['minimum_training_coverage']:
        raise ValueError('Training coverage below gate; inspect length_report, increase budget or redesign prompts explicitly')
    return encoded


def selective_loss(model, inputs, target_chunk=32):
    return completion_loss(model, inputs, target_chunk)


def student_collator(tokenizer):
    import torch
    def collate(batch):
        length = max(len(r['input_ids']) for r in batch)
        return {key: torch.tensor([r[key] + [(-100 if key == 'labels' else tokenizer.pad_token_id if key == 'input_ids' else 0)]
            * (length-len(r[key])) for r in batch], dtype=torch.long) for key in ('input_ids', 'attention_mask', 'labels')}
    return collate


def student_smoke_steps(model, batch, cfg):
    import torch
    model.train()
    torch.cuda.reset_peak_memory_stats()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable: raise ValueError('Student has no trainable adapter parameters')
    optimizer = torch.optim.AdamW(trainable, lr=cfg['student_lr'])
    scaler = torch.amp.GradScaler('cuda', init_scale=cfg.get('student_amp_init_scale', 128.))
    losses, norms = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss, outputs = selective_loss(model, batch, cfg.get('student_loss_chunk', 32))
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        norms.append(norm.item())
        if not norm: raise ValueError('No student adapter gradient; cannot certify optimizer path')
        scaler.step(optimizer); scaler.update()
        del loss, outputs
    return {'sequence_tokens': batch['input_ids'].shape[1], 'losses': losses,
            'grad_norms': norms, 'optimizer_steps': 2,
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
            'attention_query_chunk': cfg.get('attention_query_chunk',128),
            'target_loss_chunk': cfg.get('student_loss_chunk',32)}


def student_precheck(cfg):
    """Full configured training length, random probe only; no teacher/ToolEnv/data labels."""
    import torch
    out, _ = paths(cfg)
    model, tokenizer = load_llm(cfg, train=True)
    rows,catalog,split=load_data(cfg)
    fit=set(selected_ids(cfg,split,'fit')); counts=[]
    for row in rows:
        if row['id'] not in fit: continue
        for record in supervised_examples(row,schemas(row,catalog),row['label']['tool_calls'],row['label']['tool_results'],'gold'):
            encoded,n=training_tokens(tokenizer,record,cfg['student_train_context'])
            counts.append((encoded is not None,n))
    coverage=sum(ok for ok,_ in counts)/max(1,len(counts))
    write_json(out/'gold_length_precheck.json',{'examples':len(counts),'coverage':coverage,
        'max_tokens':max((n for _,n in counts),default=0),'context':cfg['student_train_context'],
        'no_truncation':True,'scope':'gold fit-only length gate before any teacher generation'})
    if coverage<cfg['minimum_training_coverage']:
        raise ValueError('Gold fit coverage below gate before teacher; inspect gold_length_precheck.json')
    length = cfg['student_train_context']
    seed_tokens = tokenizer.encode(' kiểm tra bộ nhớ ', add_special_tokens=False)
    if not seed_tokens: raise ValueError('Empty resource probe tokens')
    ids = (seed_tokens * math.ceil(length/len(seed_tokens)))[:length]
    labels = [-100]*length
    target = min(512, length-1)
    labels[-target:] = ids[-target:]
    batch = {k: torch.tensor([v],device='cuda:0') for k,v in
             {'input_ids':ids,'attention_mask':[1]*length,'labels':labels}.items()}
    report = {'purpose':'resource probe only; adapter updates are discarded',
              'full_configured_context':True, 'supervised_probe_tokens':target}
    try:
        report.update(student_smoke_steps(model,batch,cfg),status='passed')
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        write_json(out/'student_precheck.json',report)
        print('Student early precheck:',json.dumps(report),flush=True)


def stack_check(cfg):
    from verify_training_stack import run
    out, _ = paths(cfg)
    report=run(out/'stack_checks'/str(time.time_ns()))
    write_json(out/'stack_check.json',report)


def student_train(cfg, smoke=False):
    import torch
    from transformers import Trainer
    out, _ = paths(cfg)
    model, tokenizer = load_llm(cfg, train=True)
    data = tokenized_training(cfg, tokenizer)
    collate = student_collator(tokenizer)
    if smoke:
        batch = collate([max(data, key=lambda r: len(r['input_ids']))])
        batch = {k: v.to('cuda:0') for k, v in batch.items()}
        report = student_smoke_steps(model, batch, cfg)
        report['scope'] = 'longest retained example; adapter updates discarded before full train'
        write_json(out / 'student_memory_smoke.json',report)
        print('Student real-shape memory smoke passed', flush=True)
        return
    class CompletionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = selective_loss(model, inputs, cfg.get('student_loss_chunk', 32))
            return (loss, outputs) if return_outputs else loss
    trainer = CompletionTrainer(model=model, args=trainer_args(cfg, out / 'student_checkpoints'),
        train_dataset=data, data_collator=collate)
    # Default 65536 overflowed the first real NF4/LoRA backward in the compatibility
    # lab. Keep the smoke and Trainer scaler policy identical; resume restores state.
    if trainer.accelerator.scaler is not None:
        trainer.accelerator.scaler = torch.amp.GradScaler('cuda', init_scale=cfg.get('student_amp_init_scale',128.))
    trainer.model_accepts_loss_kwargs = False
    callback = deadline_callback(); trainer.add_callback(callback)
    trainer.train(resume_from_checkpoint=last_checkpoint(out / 'student_checkpoints'))
    if callback.stopped:
        raise SessionBudget('SESSION_BUDGET: student optimizer checkpoint saved; resume training next session')
    trainer.save_model(str(out / 'student_adapter')); tokenizer.save_pretrained(out / 'student_adapter')
    trainer.save_state()
    if not any(p.detach().abs().max().item() > 0 for name,p in model.named_parameters()
               if 'lora_B' in name and p.requires_grad):
        raise RuntimeError('No student LoRA B update occurred; inspect AMP/optimizer diagnostics')
    write_json(out / 'student_training.json', {'steps': trainer.state.global_step, 'examples': len(data),
        'logical_parameters': logical_parameters(model),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'loss': 'assistant completion only, selected causal positions before lm_head; mean per microbatch',
        'attention_backend': 'lifemate_bounded_'+str(cfg.get('attention_query_chunk',128)),
        'target_loss_chunk':cfg.get('student_loss_chunk',32),
        'compact_frozen_io':cfg.get('student_compact_frozen_io',True),
        'teacher_used_for_inference': False})


def agent_eval(cfg, trained=False, audio=False):
    out, root = paths(cfg)
    rows, catalog, split = load_data(cfg)
    ids = selected_ids(cfg, split, 'val')
    tag = 'agent_audio' if audio else 'agent_tuned_text' if trained else 'agent_base_text'
    target, active, records = guarded_records(out, tag, ids)
    refs = {r['id']: r for r in read_jsonl(out / 'references.jsonl')}
    model, tokenizer = load_llm(cfg, adapter=out / 'student_adapter' if trained else None)
    model.eval(); generate = generator(model, tokenizer, cfg)
    asr = processor = None
    if audio:
        selected = json.loads((out / 'selected_asr.json').read_text(encoding='utf-8'))
        asr, processor = load_asr(cfg, adapter=out / selected['adapter'] if selected['adapter'] else None)
        asr.eval()
    factory = runtime(cfg); by_id = {r['id']: r for r in rows}
    for sid in ids[len(records):]:
        budget_guard(cfg['agent_seconds'] + 120)
        write_json(active, {'id': sid, 'purpose': tag})
        started = time.monotonic(); env = factory('train', sid, root=root); transcripts = None
        try:
            if audio:
                transcripts = {}
                for _, turn in audio_refs(by_id[sid]):
                    transcripts[turn['audio']] = transcribe(asr, processor, cfg, turn['audio'], started + cfg['agent_seconds'])
            remaining = cfg['agent_seconds'] - (time.monotonic()-started)
            if remaining <= 0: raise TimeoutError('Input/ASR exhausted budget')
            result = run_agent(by_id[sid], catalog, env, generate, seconds=remaining,
                max_steps=cfg['max_steps'], max_calls=cfg['max_calls'], transcripts=transcripts)
        except Exception as exc:
            result = {'id': sid, 'tool_calls': env.get_tool_calls(), 'state_hash': env.get_state_hash(),
                'status': 'timeout' if isinstance(exc, TimeoutError) else 'asr_error', 'error': type(exc).__name__}
        result['elapsed_s'] = time.monotonic()-started
        if result['elapsed_s'] > cfg['agent_seconds']: result['status'] = 'timeout'
        result['local_score'] = score(result, refs[sid], catalog, failed=failed_status(result['status']))
        append_jsonl(target, result); active.unlink(); records.append(result)
        print(tag, len(records), '/', len(ids), result['status'], flush=True)
    times = sorted(r['elapsed_s'] for r in records)
    summary = {'samples': len(records), 'ids': ids, 'mode': cfg['mode'],
        'score_spec_local': statistics.mean(r['local_score']['score_spec_local'] for r in records),
        'state_match': statistics.mean(r['local_score']['state_match'] for r in records),
        'tool_f1': statistics.mean(r['local_score']['tool_f1'] for r in records),
        'statuses': dict(Counter(r['status'] for r in records)),
        'p50_s': statistics.median(times), 'p95_s': times[math.ceil(len(times)*.95)-1], 'max_s': max(times),
        'scorer': SCORER_VERSION, 'not_official_score': True,
        'timing': 'live ASR current/history + one env + LLM/tools' if audio else 'text + one env + LLM/tools',
        'timing_limit': 'local Kaggle diagnostic; synchronous kernels may overrun; startup and result serialization logged separately/not certified BTC timing'}
    write_json(out / (tag + '_metrics.json'), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'ids'}), flush=True)


def export(cfg):
    out, _ = paths(cfg)
    selected = json.loads((out / 'selected_asr.json').read_text(encoding='utf-8'))
    student = json.loads((out / 'student_training.json').read_text(encoding='utf-8'))
    total = selected['logical_parameters'] + student['logical_parameters']
    reuse = json.loads((out / 'asr_reuse.json').read_text(encoding='utf-8'))
    teacher_reuse = json.loads((out/'teacher_reuse.json').read_text(encoding='utf-8'))
    manifest = {'mode': cfg['mode'], 'student': {'id': cfg['student_id'], 'revision': cfg['student_revision'], 'adapter': 'student_adapter'},
        'asr': {'id': cfg['asr_id'], 'revision': cfg['asr_revision'], 'adapter': selected['adapter']},
        'teacher_development_only': {'id': cfg['teacher_id'], 'revision': cfg['teacher_revision']},
        'asr_artifact_origin': reuse.get('source_run') if reuse['enabled'] else 'current_run',
        'teacher_data_origin': teacher_reuse.get('source_run') if teacher_reuse['enabled'] else 'current_run',
        'attention_backend': 'lifemate_bounded_'+str(cfg.get('attention_query_chunk',128)),
        'total_inference_logical_parameters': total, 'llm_logical_parameters': student['logical_parameters'],
        'parameter_gate_passed': student['logical_parameters'] < 5_000_000_000 and total < 6_000_000_000,
        'not_official_submission': True, 'base_weights_required_for_offline_inference': True,
        'artifacts': {str(p.relative_to(out)): digest(p) for name in ('asr_adapter', 'student_adapter')
            for p in sorted((out / name).rglob('*')) if p.is_file()},
        'limitations': ['pilot results are not final/full validation', 'hardware/CUDA rules require latest BTC confirmation',
            'quantized logical count is diagnostic; confirm organizer parameter accounting',
            'offline image must include base weights; this export contains adapters/tokenizers/manifests']}
    write_json(out / 'export_manifest.json', manifest)
    report = {'mode': cfg['mode'], 'parameter_gate_passed': manifest['parameter_gate_passed'],
              'asr_reuse': {k: v for k, v in reuse.items() if k in ('enabled', 'source_run', 'source_identity', 'note')},
              'teacher_reuse': teacher_reuse}
    for name in ('stack_check','student_precheck','data_report', 'asr_base_metrics', 'asr_tuned_metrics', 'teacher_metrics', 'examples_report',
                 'length_report', 'student_memory_smoke', 'agent_base_text_metrics', 'agent_tuned_text_metrics', 'agent_audio_metrics'):
        obj = json.loads((out / (name + '.json')).read_text(encoding='utf-8'))
        report[name] = {k: v for k, v in obj.items() if k not in ('ids', 'rejected')}
    write_json(out / 'FINAL_REPORT.json', report)
    if not manifest['parameter_gate_passed']:
        raise ValueError('Inference parameter budget failed')
    print('Export ready:', out, '| mode:', cfg['mode'], '| total params:', total, flush=True)


STAGES = {'preflight': preflight, 'stack_check': stack_check, 'prepare': prepare, 'reuse_asr': reuse_asr, 'references': references,
    'reuse_teacher': reuse_teacher, 'student_precheck': student_precheck,
    'asr_base': lambda c: asr_eval(c), 'asr_train': asr_train,
    'asr_tuned': lambda c: asr_eval(c, trained=True), 'teacher': teacher,
    'examples': make_examples, 'student_base': lambda c: agent_eval(c),
    'student_smoke': lambda c: student_train(c, smoke=True), 'student_train': student_train,
    'student_text': lambda c: agent_eval(c, trained=True),
    'student_audio': lambda c: agent_eval(c, trained=True, audio=True), 'export': export}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    out = Path(cfg['run_dir']); out.mkdir(parents=True, exist_ok=True)
    marker = out / ('stage_' + args.stage + '.json')
    identity = fingerprint(cfg)
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old['identity'] != identity:
            raise ValueError('Config changed inside existing run; use a new RUN_NAME')
        if old['status'] == 'completed':
            for name, value in old['artifacts'].items():
                if not (out / name).is_file() or digest(out / name) != value:
                    raise ValueError('Completed stage artifact missing/changed; do not trust marker')
            if args.stage != 'preflight':
                print('RESUME: verified completed stage', args.stage, flush=True)
                return
    start = time.monotonic()
    write_json(marker, {'identity': identity, 'stage': args.stage, 'status': 'running'})
    try:
        budget_guard(300)
        STAGES[args.stage](cfg)
    except BaseException as exc:
        write_json(marker, {'identity': identity, 'stage': args.stage, 'status': 'failed',
                           'error': type(exc).__name__ + ': ' + str(exc), 'elapsed_s': time.monotonic()-start})
        if isinstance(exc, SessionBudget):
            print(str(exc), flush=True)
            return 75
        raise
    # Stage-specific outputs are immutable after completion; shared provenance
    # is excluded because later stages legitimately create new artifacts.
    outputs = {'preflight': ['provenance.json'], 'prepare': ['split.json', 'asr_fit.json', 'asr_val.json', 'data_report.json'],
        'reuse_asr': ['asr_reuse.json'],
        'reuse_teacher': ['teacher_reuse.json'], 'student_precheck': ['student_precheck.json','gold_length_precheck.json'],
        'stack_check': ['stack_check.json'],
        'references': ['references.jsonl'], 'asr_base': ['asr_base_eval.jsonl', 'asr_base_metrics.json'],
        'asr_train': ['asr_training.json'], 'asr_tuned': ['asr_tuned_eval.jsonl', 'asr_tuned_metrics.json', 'selected_asr.json'],
        'teacher': ['teacher_rollouts.jsonl', 'teacher_metrics.json'], 'examples': ['student_examples.jsonl', 'examples_report.json'],
        'student_base': ['agent_base_text.jsonl', 'agent_base_text_metrics.json'],
        'student_smoke': ['student_memory_smoke.json'], 'student_train': ['student_training.json'],
        'student_text': ['agent_tuned_text.jsonl', 'agent_tuned_text_metrics.json'],
        'student_audio': ['agent_audio.jsonl', 'agent_audio_metrics.json'], 'export': ['export_manifest.json', 'FINAL_REPORT.json']}[args.stage]
    if args.stage in ('asr_train', 'student_train'):
        outputs += [str(p.relative_to(out)) for p in (out / ('asr_adapter' if args.stage == 'asr_train' else 'student_adapter')).rglob('*') if p.is_file()]
    write_json(marker, {'identity': identity, 'stage': args.stage, 'status': 'completed',
                       'elapsed_s': time.monotonic()-start, 'artifacts': {name: digest(out / name) for name in outputs}})


if __name__ == '__main__':
    # The embedded stack check imports worker; preserve exception class identity
    # so SessionBudget remains a clean code-75 pause rather than a generic failure.
    sys.modules.setdefault('worker',sys.modules[__name__])
    sys.exit(main())
