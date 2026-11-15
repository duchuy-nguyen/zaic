"""Component-specific source reference extracted from the original successful trainer.
No CLI, model loading, data preparation, or training occurs on import.
Requires restored private manifests/data/snapshots; not a newly validated training CLI.
"""
from contextlib import nullcontext
import gc,hashlib,importlib.metadata,json,math,os,random,sys,time
from pathlib import Path
from .artifact_io import sha,new,json_new
from .tokenization import encode
from .completion_loss import collate_rows,completion_causal_loss
HERE=ROOT=LLM=LIFE=None

def bind_private_workspace(path, workspace_root):
    global HERE,ROOT,LLM,LIFE
    folder=Path(path).resolve();root=Path(workspace_root).resolve()
    if not folder.is_dir() or not root.is_dir():
        raise ValueError("Private training workspace paths must already exist")
    config=json.loads((folder/"config.json").read_text(encoding="utf8"))
    if config["llm"]["base"]!="Qwen/Qwen3.5-4B": raise ValueError("Expected original Qwen3.5 V2 workspace")
    HERE=folder;ROOT=root;LLM=root/"llm";LIFE=root/"LifeMate"

from .qwen_training_policy import words,distance

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def atomic(p, d):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    t.replace(p)

def verify(data, full=False):
    d = read(data / 'DATA_MANIFEST.json')
    for n in ('train', 'val'):
        assert sha(data / (n + '.jsonl')) == d['partitions'][n]['sha256']
    assert len(d['partitions']['train']['source_ids']) == 1236 and len(d['partitions']['val']['source_ids']) == 309
    assert not set(d['partitions']['train']['source_ids']) & set(d['partitions']['val']['source_ids'])
    for n, h in d['feature_source_sha256'].items():
        assert sha(HERE / n) == h, n
    if full:
        p = read(HERE / 'PACKAGE_MANIFEST.json')
        for n, h in p['files'].items():
            assert sha(HERE / n) == h, n
        assert p['data_manifest_sha256'] == sha(data / 'DATA_MANIFEST.json')
    return d

def model_snapshot():
    return Path(read(HERE / '.runtime/MODELS.json')['snapshot'])

def profile(data, output):
    verify(data)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot(), local_files_only=True)
    result = {'partitions': {}, 'no_truncation': True}
    for part in ('train', 'val'):
        source = [json.loads(x) for x in (data / (part + '.jsonl')).read_text(encoding='utf8').splitlines()]
        encoded = [encode(e, tokenizer) for e in source]
        lengths = sorted((r['length'] for r in encoded))
        new(output / (part + '_tokens.jsonl'), ''.join((json.dumps(x, separators=(',', ':')) + '\n' for x in encoded)))
        result['partitions'][part] = {'n': len(encoded), 'max_length': max(lengths), 'p50': statistics_median(lengths), 'p95': lengths[math.ceil(0.95 * len(lengths)) - 1], 'supervised_tokens': sum((x['target_tokens'] for x in encoded)), 'token_cache_sha256': sha(output / (part + '_tokens.jsonl')), 'longest_train_ids': [x['id'] for x in sorted(encoded, key=lambda r: r['length'], reverse=True)[:2]]}
    json_new(output / 'TOKEN_PROFILE.json', result)
    print('TOKEN_PROFILE', json.dumps(result), flush=True)

def statistics_median(values):
    import statistics
    return statistics.median(values)

def gpu_setup():
    import numpy as np
    import torch
    random.seed(20261103)
    np.random.seed(20261103)
    torch.manual_seed(20261103)
    torch.cuda.manual_seed_all(20261103)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported(), 'Local BF16 training needs a natively supported GPU'
    torch.cuda.set_per_process_memory_fraction(0.95, 0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return (torch, np)

def training(data, output, smoke, resume):
    if HERE is None:
        raise RuntimeError('Call bind_private_workspace with the restored private training workspace first')
    manifest = verify(data, full=not smoke)
    cfg = manifest['config']['llm']
    torch, np = gpu_setup()
    from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from transformers.trainer_utils import get_last_checkpoint
    output.mkdir(parents=True, exist_ok=True)
    run = output / ('llm' + ('_smoke' if smoke else '_full'))
    run.mkdir(exist_ok=True)
    previous = read(run / 'RUN_MANIFEST.json') if (run / 'RUN_MANIFEST.json').exists() else None
    common = {'data_sha256': sha(data / 'DATA_MANIFEST.json'), 'source_sha256': sha(Path(__file__)), 'kind': 'llm', 'smoke': smoke, 'config': cfg, 'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')}}
    if previous:
        assert previous['configuration'] == common, 'Training configuration changed'
        if previous['status'] == 'completed':
            print('ALREADY_COMPLETED', 'llm', smoke, flush=True)
            return
        if not resume:
            raise RuntimeError('Interrupted run exists; explicitly resume from last saved optimizer checkpoint')
    else:
        atomic(run / 'RUN_MANIFEST.json', {'status': 'preparing', 'configuration': common})
    checkpoint = get_last_checkpoint(str(run / 'checkpoints')) if resume and (run / 'checkpoints').exists() else None
    if resume and previous and (not checkpoint):
        raise RuntimeError('No saved training checkpoint; preserve interrupted artifacts')
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from .efficient_attention import install, probe
    probe()
    install()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_snapshot(), local_files_only=True, quantization_config=quant, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='sdpa')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
    for name, p in model.named_parameters():
        if not p.requires_grad and p.is_floating_point() and (not isinstance(p, bnb.nn.Params4bit)) and (not any((x in name for x in ('norm', 'A_log', 'dt_bias')))):
            p.data = p.data.to(torch.bfloat16)
    targets = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and 'language_model' in n and (not any((x in n for x in ('lm_head', 'embed_tokens', 'visual'))))]
    assert targets and any(('linear_attn' in n for n in targets))
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'], target_modules=targets, bias='none'))
    model.peft_config['default'].base_model_name_or_path = cfg['base']
    model.peft_config['default'].revision = cfg['revision']
    processor = AutoTokenizer.from_pretrained(model_snapshot(), local_files_only=True)
    prof = read(output / 'TOKEN_PROFILE.json')
    assert sha(output / 'train_tokens.jsonl') == prof['partitions']['train']['token_cache_sha256']
    train_rows = [json.loads(l) for l in (output / 'train_tokens.jsonl').read_text(encoding='utf8').splitlines()]
    if smoke:
        train_rows = sorted(train_rows, key=lambda r: r['length'], reverse=True)[:2]
    eval_rows = None if smoke else [json.loads(l) for l in (output / 'val_tokens.jsonl').read_text(encoding='utf8').splitlines()]
    collator = lambda rows: collate_rows(rows, processor.pad_token_id, torch)
    model.config.use_cache = False
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(('lora_' in n and 'visual' not in n for n, p in trainable))
    params = sum((p.numel() for n, p in trainable))
    initial_b = {n: p.detach().cpu().clone() for n, p in trainable if '.lora_B.' in n}
    args = dict(output_dir=str(run / 'checkpoints'), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=1 if smoke else cfg['gradient_accumulation'], num_train_epochs=cfg['epochs'], learning_rate=cfg['learning_rate'], warmup_steps=0.05, weight_decay=0.01, lr_scheduler_type='linear', bf16=True, fp16=False, optim='adamw_torch', gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=1, logging_nan_inf_filter=False, torch_empty_cache_steps=1, save_steps=1 if smoke else 10, save_strategy='steps', save_total_limit=2, eval_strategy='no', remove_unused_columns=False, dataloader_num_workers=0, report_to='none', seed=20261103, data_seed=20261103, max_grad_norm=1.0, push_to_hub=False, prediction_loss_only=True)
    if smoke:
        args['max_steps'] = 2
    training_args = TrainingArguments(**args)
    started = time.monotonic()
    val_history = read(run / 'VAL_HISTORY.json') if resume and (run / 'VAL_HISTORY.json').exists() else []

    class Progress(TrainerCallback):

        def on_step_end(self, args, state, control, **kwargs):
            atomic(run / 'PROGRESS.json', {'step': state.global_step, 'total_steps': state.max_steps, 'elapsed_s': time.monotonic() - started, 'cuda_max_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
            if state.global_step >= state.max_steps:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            return control
    from .completion_loss import completion_causal_loss

    class CompletionTrainer(Trainer):

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = completion_causal_loss(model, inputs, num_items_in_batch)
            return (loss, outputs) if return_outputs else loss
    trainer = CompletionTrainer(model=model, args=training_args, train_dataset=train_rows, eval_dataset=eval_rows, data_collator=collator, callbacks=[Progress()])
    atomic(run / 'RUN_MANIFEST.json', {'status': 'running', 'configuration': common, 'trainable_parameters': params})
    print('TRAIN_START', 'llm', 'smoke', smoke, 'examples', len(train_rows), 'trainable', params, 'GPU', torch.cuda.get_device_name(0), flush=True)
    try:
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            result = trainer.train(resume_from_checkpoint=checkpoint)
        final = run / 'adapter_final'
        model.save_pretrained(final, safe_serialization=True)
        processor.save_pretrained(run / 'processor')
        tensors = load_file(str(final / 'adapter_model.safetensors'))
        assert all((torch.isfinite(t).all().item() for t in tensors.values()))
        changed = any((not torch.equal(initial_b[n], p.detach().cpu()) for n, p in model.named_parameters() if n in initial_b))
        if checkpoint and (not changed):
            changed = any((torch.count_nonzero(t).item() > 0 for n, t in tensors.items() if '.lora_B.' in n))
        assert changed, 'LoRA matrices did not change'
        val_loss = trainer.evaluate() if True and (not smoke) else None
        atomic(run / 'RUN_MANIFEST.json', {'status': 'completed', 'configuration': common, 'trainable_parameters': params, 'step': trainer.state.global_step, 'total_steps': trainer.state.max_steps, 'adapter': str(final), 'adapter_changed': changed, 'adapter_finite': True, 'training_metrics': result.metrics, 'val_loss': val_loss, 'asr_val_history': val_history, 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
        print('TRAIN_COMPLETED', 'llm', smoke, final, flush=True)
    except BaseException as e:
        atomic(run / 'RUN_MANIFEST.json', {'status': 'failed_or_interrupted', 'configuration': common, 'error': repr(e), 'trainable_parameters': params, 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
        raise
