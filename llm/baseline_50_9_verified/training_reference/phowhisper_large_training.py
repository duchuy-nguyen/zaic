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
    recipe=json.loads((folder/"recipe.json").read_text(encoding="utf8"))
    if recipe["asr"]["base"]!="vinai/PhoWhisper-large": raise ValueError("Expected original PhoWhisper-large workspace")
    HERE=folder;ROOT=root;LLM=root/"llm";LIFE=root/"LifeMate"

from .artifact_io import read,atomic
from .asr_training_policy import words,distance

def verify_data():
    parity = read(HERE / 'DATA_PARITY.json')
    for n, h in parity['data_files'].items():
        assert sha(HERE / 'data' / n) == h, n
    for n, h in parity['unchanged_baselines'].items():
        assert sha(n) == h, n
    split = read(HERE / 'data/split.json')
    assert len(split['train_fit']) == 1236 and len(split['val_main']) == 309
    assert not set(split['train_fit']) & set(split['val_main'])
    assert not {split['group_ids'][i] for i in split['train_fit']} & {split['group_ids'][i] for i in split['val_main']}
    train = read(HERE / 'data/train_asr_clips.json')
    val = read(HERE / 'data/val_asr_clips.json')
    assert len(train) == 3997 and len(val) == 1039
    assert not {r['sha256'] for r in train} & {r['sha256'] for r in val}
    return read(HERE / 'recipe.json')

def snapshot(kind):
    return Path(read(HERE / '.runtime/MODELS.json')['asr']['snapshot'])

def verify_frozen():
    p = HERE / 'FROZEN_TRAINING_SOURCE.json'
    if p.exists():
        for n, h in read(p)['files_sha256'].items():
            assert sha(HERE / n) == h, n

def gpu_setup():
    import numpy as np, torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported()
    random.seed(20261103)
    np.random.seed(20261103)
    torch.manual_seed(20261103)
    torch.cuda.manual_seed_all(20261103)
    free, total = torch.cuda.mem_get_info()
    fraction = min(0.95, max(0.1, (free - 256 * 1024 ** 2) / total))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print('PHYSICAL_VRAM_CAP', fraction, 'free_bytes', free, flush=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return (torch, np)

def asr_dataset(clips, processor, torch, np):
    import wave
    base = ROOT / 'LifeMate/data/lifemate-v3.2.0/train+public_test/train'

    class Dataset(torch.utils.data.Dataset):

        def __len__(self):
            return len(clips)

        def __getitem__(self, index):
            r = clips[index]
            p = (base / r['audio']).resolve()
            assert p.is_relative_to((base / 'audio').resolve())
            with wave.open(str(p), 'rb') as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
                values = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768
            features = processor.feature_extractor(values, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            return {'input_features': features.input_features[0], 'attention_mask': features.attention_mask[0], 'labels': processor.tokenizer(r['text']).input_ids}
    return Dataset()

def asr_collator(proc, decoder_start, torch):

    def collate(rows):
        lab = proc.tokenizer.pad([{'input_ids': r['labels']} for r in rows], return_tensors='pt')
        y = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
        if (y[:, 0] == decoder_start).all().item():
            y = y[:, 1:]
        return {'input_features': torch.stack([r['input_features'] for r in rows]), 'attention_mask': torch.stack([r['attention_mask'] for r in rows]), 'labels': y}
    return collate

def evaluate_asr(model, proc, torch, np, run, step):
    destination = run / f'VAL_ASR_STEP_{step}.json'
    if destination.exists():
        return read(destination)
    clips = read(HERE / 'data/val_asr_clips.json')
    ds = asr_dataset(clips, proc, torch, np)
    training = model.training
    model.eval()
    rows = []
    edits = den = failed = 0
    started = time.monotonic()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for j, r in enumerate(clips):
            x = ds[j]
            out = model.generate(input_features=x['input_features'][None].to('cuda', dtype=torch.bfloat16), attention_mask=x['attention_mask'][None].to('cuda'), language='vi', task='transcribe', do_sample=False, num_beams=1, max_new_tokens=256, return_timestamps=False, return_dict_in_generate=True, use_cache=True)
            capped = not bool(eos.intersection(out.sequences[0].tolist()))
            text = '' if capped else proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
            a, b = (words(r['text']), words(text))
            edits += distance(a, b)
            den += len(a)
            failed += int(capped)
            rows.append({'audio': r['audio'], 'sha256': r['sha256'], 'text': text, 'error': 'token_cap_without_eos' if capped else None})
            if (j + 1) % 100 == 0:
                print('ASR_VAL', step, j + 1, '/', len(clips), flush=True)
    if training:
        model.train()
    result = {'step': step, 'n': len(clips), 'wer_percent': 100 * edits / den, 'edits': edits, 'reference_words': den, 'seconds': time.monotonic() - started, 'failed_samples': failed, 'predictions': rows}
    json_new(destination, result)
    return result

def training(smoke, resume, smoke_tag='v1'):
    if HERE is None:
        raise RuntimeError('Call bind_private_workspace with the restored private training workspace first')
    recipe = verify_data()
    verify_frozen()
    cfg = recipe['asr']
    torch, np = gpu_setup()
    from transformers import Trainer, TrainingArguments, TrainerCallback, WhisperForConditionalGeneration, WhisperProcessor
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
    from safetensors.torch import load_file
    run = HERE / 'outputs' / ('asr' + ('_smoke_' + smoke_tag if smoke else '_full'))
    run.mkdir(parents=True, exist_ok=True)
    common = {'data_parity_sha256': sha(HERE / 'DATA_PARITY.json'), 'recipe_sha256': sha(HERE / 'recipe.json'), 'source_sha256': sha(Path(__file__)), 'kind': 'asr', 'smoke': smoke, 'config': cfg, 'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')}}
    marker = run / 'RUN_MANIFEST.json'
    previous = read(marker) if marker.exists() else None
    if previous:
        assert previous['configuration'] == common, 'Training source/config changed; preserve checkpoint and version explicitly'
        if previous['status'] == 'completed':
            print('ALREADY_COMPLETED', 'asr', smoke, flush=True)
            return
        assert resume, 'Explicit --resume required'
    else:
        atomic(marker, {'status': 'preparing', 'configuration': common})
    checkpoint = get_last_checkpoint(str(run / 'checkpoints')) if resume and (run / 'checkpoints').exists() else None
    if resume and previous and (not checkpoint):
        raise RuntimeError('No complete checkpoint; preserve artifacts and investigate')
    proc = WhisperProcessor.from_pretrained(snapshot('asr'), local_files_only=True)
    proc.tokenizer.set_prefix_tokens(language='vi', task='transcribe', predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(snapshot('asr'), local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    assert sum((p.numel() for p in model.parameters())) == cfg['expected_base_parameters']
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = 'vi'
    model.generation_config.task = 'transcribe'
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'], target_modules=cfg['targets'], bias='none'))
    clips = read(HERE / 'data/train_asr_clips.json')
    if smoke:
        clips = sorted(clips, key=lambda r: len(words(r['text'])), reverse=True)[:2]
    train_rows = asr_dataset(clips, proc, torch, np)
    collator = asr_collator(proc, model.config.decoder_start_token_id, torch)
    eval_rows = None
    model.peft_config['default'].base_model_name_or_path = cfg['base']
    model.peft_config['default'].revision = cfg['revision']
    model.config.use_cache = False
    trained = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trained and all(('lora_' in n for n, p in trained))
    params = sum((p.numel() for n, p in trained))
    initial_b = {n: p.detach().cpu().clone() for n, p in trained if '.lora_B.' in n}
    actual = recipe['asr' + '_actual_trainer']
    args = dict(output_dir=str(run / 'checkpoints'), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=1 if smoke else actual['gradient_accumulation'], num_train_epochs=cfg['epochs'], learning_rate=cfg['learning_rate'], warmup_steps=actual['warmup_steps'], weight_decay=actual['weight_decay'], lr_scheduler_type='linear', bf16=True, fp16=False, optim='adamw_torch', gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=1, logging_nan_inf_filter=False, torch_empty_cache_steps=None, save_steps=1 if smoke else 10, save_strategy='steps', save_total_limit=2, eval_strategy='no', remove_unused_columns=False, dataloader_num_workers=0, report_to='none', seed=20261103, data_seed=20261103, max_grad_norm=1.0, push_to_hub=False, prediction_loss_only=True)
    if smoke:
        args['max_steps'] = 2
    training_args = TrainingArguments(**args)
    started = time.monotonic()
    val_history = read(run / 'VAL_HISTORY.json') if (run / 'VAL_HISTORY.json').exists() else []
    session_start_step = previous.get('last_step', 0) if previous else 0

    class Progress(TrainerCallback):

        def on_step_end(self, args, state, control, **kwargs):
            atomic(run / 'PROGRESS.json', {'step': state.global_step, 'total_steps': state.max_steps, 'elapsed_s': time.monotonic() - started, 'cuda_max_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
            if state.global_step >= state.max_steps:
                control.should_save = True
            if (HERE / 'PAUSE_REQUESTED').exists():
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            saved = run / 'checkpoints' / f'checkpoint-{state.global_step}'
            required = ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth']
            assert all(((saved / n).is_file() for n in required)), str(saved)
            atomic(run / 'LATEST_CHECKPOINT.json', {'step': state.global_step, 'total_steps': state.max_steps, 'path': str(saved), 'files_sha256': {n: sha(saved / n) for n in required}})
            if True and (not smoke) and (abs(state.epoch - round(state.epoch)) < 1e-06 or state.global_step >= state.max_steps):
                epoch = run / 'epoch_adapters' / f'step-{state.global_step}'
                if not epoch.exists():
                    model.save_pretrained(epoch, safe_serialization=True)
                    proc.save_pretrained(epoch / 'processor')
                    json_new(epoch / 'WEIGHTS_MANIFEST.json', {'step': state.global_step, 'epoch': state.epoch, 'adapter_sha256': sha(epoch / 'adapter_model.safetensors')})
                if not any((v['step'] == state.global_step for v in val_history)):
                    result = evaluate_asr(model, proc, torch, np, run, state.global_step)
                    val_history.append({'step': state.global_step, 'wer_percent': result['wer_percent'], 'checkpoint': str(saved)})
                    if min(val_history, key=lambda v: v['wer_percent'])['step'] == state.global_step:
                        model.save_pretrained(run / 'best_adapter', safe_serialization=True)
                        proc.save_pretrained(run / 'processor')
                    atomic(run / 'VAL_HISTORY.json', val_history)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ('loss', 'grad_norm', 'eval_loss'):
                if logs and key in logs and (not math.isfinite(float(logs[key]))):
                    raise FloatingPointError('Nonfinite ' + key)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_rows, data_collator=collator, callbacks=[Progress()])
    trainer.model_accepts_loss_kwargs = False
    atomic(marker, {'status': 'running', 'configuration': common, 'trainable_parameters': params})
    print('TRAIN_START', 'asr', 'smoke', smoke, 'examples', len(train_rows), 'trainable', params, 'GPU', torch.cuda.get_device_name(0), flush=True)
    try:
        with nullcontext():
            result = trainer.train(resume_from_checkpoint=checkpoint)
        if trainer.state.global_step < trainer.state.max_steps:
            atomic(marker, {'status': 'paused_by_user', 'configuration': common, 'last_step': trainer.state.global_step, 'total_steps': trainer.state.max_steps, 'trainable_parameters': params})
            print('PAUSED_SAFELY', 'asr', trainer.state.global_step, flush=True)
            return
        final = run / 'adapter_final'
        model.save_pretrained(final, safe_serialization=True)
        proc.save_pretrained(run / 'processor')
        tensors = load_file(str(final / 'adapter_model.safetensors'))
        assert all((torch.isfinite(t).all().item() for t in tensors.values()))
        changed = any((not torch.equal(initial_b[n], p.detach().cpu()) for n, p in model.named_parameters() if n in initial_b))
        if checkpoint and (not changed):
            changed = any((torch.count_nonzero(t).item() > 0 for n, t in tensors.items() if '.lora_B.' in n))
        assert changed
        current = get_peft_model_state_dict(model)
        assert set(current) == set(tensors) and all((torch.equal(current[n].detach().cpu(), t) for n, t in tensors.items()))
        json_new(run / 'WEIGHTS_VERIFIED.json', {'finite': True, 'changed': True, 'exact_readback': True, 'adapter_sha256': sha(final / 'adapter_model.safetensors')})
        val_loss = None
        expected = 2 if smoke else 250
        assert trainer.state.global_step == trainer.state.max_steps == expected
        atomic(marker, {'status': 'completed', 'configuration': common, 'step': trainer.state.global_step, 'total_steps': trainer.state.max_steps, 'trainable_parameters': params, 'adapter': str(final), 'training_metrics': result.metrics, 'val_loss': val_loss, 'asr_val_history': val_history, 'adapter_changed': True, 'adapter_finite': True, 'cuda_max_reserved_bytes': torch.cuda.max_memory_reserved()})
        print('TRAIN_COMPLETED', 'asr', smoke, final, flush=True)
    except BaseException as error:
        atomic(marker, {'status': 'failed_or_interrupted', 'configuration': common, 'error': repr(error), 'last_step': trainer.state.global_step, 'total_steps': trainer.state.max_steps, 'trainable_parameters': params})
        raise
