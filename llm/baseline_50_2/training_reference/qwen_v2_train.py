"""Local combined V2 training stages. No public inference or upload in this worker."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time

HERE=Path(__file__).resolve().parent
LLM=HERE.parents[1];LIFE=LLM.parent/'LifeMate'
sys.path.insert(0,str(LLM/'finetune'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('WANDB_DISABLED','true')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128,garbage_collection_threshold:0.8')
from prepare import sha,new,json_new
from sft_data import encode
from train_lora import collate_rows
from policy import words,distance


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def atomic(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)


def verify(data,full=False):
    d=read(data/'DATA_MANIFEST.json')
    for n in ('train','val'):
        assert sha(data/(n+'.jsonl'))==d['partitions'][n]['sha256']
    assert len(d['partitions']['train']['source_ids'])==1236 and len(d['partitions']['val']['source_ids'])==309
    assert not set(d['partitions']['train']['source_ids'])&set(d['partitions']['val']['source_ids'])
    for n,h in d['feature_source_sha256'].items():assert sha(HERE/n)==h,n
    if full:
        p=read(HERE/'PACKAGE_MANIFEST.json')
        for n,h in p['files'].items():assert sha(HERE/n)==h,n
        assert p['data_manifest_sha256']==sha(data/'DATA_MANIFEST.json')
    return d


def model_snapshot():return Path(read(HERE/'.runtime/MODELS.json')['snapshot'])


def profile(data,output):
    verify(data)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_snapshot(),local_files_only=True)
    result={'partitions':{},'no_truncation':True}
    for part in ('train','val'):
        source=[json.loads(x) for x in (data/(part+'.jsonl')).read_text(encoding='utf8').splitlines()]
        encoded=[encode(e,tokenizer) for e in source]
        lengths=sorted(r['length'] for r in encoded)
        new(output/(part+'_tokens.jsonl'),''.join(json.dumps(x,separators=(',',':'))+'\n' for x in encoded))
        result['partitions'][part]={'n':len(encoded),'max_length':max(lengths),'p50':statistics_median(lengths),
            'p95':lengths[math.ceil(.95*len(lengths))-1],'supervised_tokens':sum(x['target_tokens'] for x in encoded),
            'token_cache_sha256':sha(output/(part+'_tokens.jsonl')),
            'longest_train_ids':[x['id'] for x in sorted(encoded,key=lambda r:r['length'],reverse=True)[:2]]}
    json_new(output/'TOKEN_PROFILE.json',result)
    print('TOKEN_PROFILE',json.dumps(result),flush=True)


def statistics_median(values):
    import statistics
    return statistics.median(values)


def gpu_setup():
    import numpy as np
    import torch
    random.seed(20261103);np.random.seed(20261103);torch.manual_seed(20261103);torch.cuda.manual_seed_all(20261103)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert torch.cuda.is_bf16_supported(),'Local BF16 training needs a natively supported GPU'
    # Fail visibly before the Windows driver silently oversubscribes physical VRAM.
    torch.cuda.set_per_process_memory_fraction(.95,0)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    return torch,np


def asr_collator(processor,decoder_start,torch):
    def collate(rows):
        labels=processor.tokenizer.pad([{'input_ids':r['labels']} for r in rows],return_tensors='pt')
        y=labels['input_ids'].masked_fill(labels.attention_mask.ne(1),-100)
        if (y[:,0]==decoder_start).all().item():y=y[:,1:]
        return {'input_features':torch.stack([r['input_features'] for r in rows]),
                'attention_mask':torch.stack([r['attention_mask'] for r in rows]),'labels':y}
    return collate


def asr_dataset(clips,processor,torch,np):
    import wave
    base=LIFE/'data/lifemate-v3.2.0/train+public_test/train'
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):return len(clips)
        def __getitem__(self,index):
            row=clips[index];p=(base/row['audio']).resolve();assert p.is_relative_to((base/'audio').resolve())
            with wave.open(str(p),'rb') as w:
                assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,16000)
                values=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
            features=processor.feature_extractor(values,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            return {'input_features':features.input_features[0], 'attention_mask':features.attention_mask[0],
                    'labels':processor.tokenizer(row['text']).input_ids}
    return Dataset()


def evaluate_asr(model,processor,clips,torch,np,output,step):
    ds=asr_dataset(clips,processor,torch,np);was_training=model.training;model.eval()
    predictions=[];edits=denominator=failed=0;started=time.monotonic()
    eos=model.generation_config.eos_token_id;eos_set=set(eos if isinstance(eos,list) else [eos])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for j,row in enumerate(clips):
            item=ds[j]
            generated=model.generate(input_features=item['input_features'][None].to('cuda',dtype=torch.bfloat16),
                attention_mask=item['attention_mask'][None].to('cuda'),language='vi',task='transcribe',
                do_sample=False,num_beams=1,max_new_tokens=256,return_timestamps=False,
                return_dict_in_generate=True,use_cache=True)
            ids=generated.sequences[0].tolist();truncated=not bool(eos_set.intersection(ids))
            # Ordinary token-cap failures remain in the WER denominator; never accept a partial transcript.
            text='' if truncated else processor.batch_decode(generated.sequences,skip_special_tokens=True)[0]
            failed+=int(truncated)
            a,b=words(row['text']),words(text);edits+=distance(a,b);denominator+=len(a)
            predictions.append({'audio':row['audio'],'sha256':row['sha256'],'text':text,'error':'token_cap_without_eos' if truncated else None})
            if (j+1)%100==0:print('ASR_VAL',step,j+1,'/',len(clips),flush=True)
    if was_training:model.train()
    result={'step':step,'n':len(clips),'wer_percent':100*edits/denominator,'edits':edits,'reference_words':denominator,
            'seconds':time.monotonic()-started,'failed_samples':failed,'predictions':predictions}
    json_new(output/f'VAL_ASR_STEP_{step}.json',result)
    return result


def training(data,output,kind,smoke,resume):
    manifest=verify(data,full=not smoke);cfg=manifest['config'][kind]
    torch,np=gpu_setup()
    from transformers import AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,WhisperForConditionalGeneration,WhisperProcessor
    from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from transformers.trainer_utils import get_last_checkpoint
    output.mkdir(parents=True,exist_ok=True)
    run=output/(kind+('_smoke' if smoke else '_full'));run.mkdir(exist_ok=True)
    previous=read(run/'RUN_MANIFEST.json') if (run/'RUN_MANIFEST.json').exists() else None
    common={'data_sha256':sha(data/'DATA_MANIFEST.json'),'source_sha256':sha(Path(__file__)),
            'kind':kind,'smoke':smoke,'config':cfg,'gpu':torch.cuda.get_device_name(0),'dtype':'bfloat16',
            'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','accelerate','bitsandbytes')}}
    if previous:
        assert previous['configuration']==common,'Training configuration changed'
        if previous['status']=='completed':print('ALREADY_COMPLETED',kind,smoke,flush=True);return
        if not resume:raise RuntimeError('Interrupted run exists; explicitly resume from last saved optimizer checkpoint')
    else:atomic(run/'RUN_MANIFEST.json',{'status':'preparing','configuration':common})
    checkpoint=get_last_checkpoint(str(run/'checkpoints')) if resume and (run/'checkpoints').exists() else None
    if resume and previous and not checkpoint:raise RuntimeError('No saved training checkpoint; preserve interrupted artifacts')
    if kind=='asr':
        folder=HERE.parent/'gipformer_val_20261106/.runtime/models/medium'
        processor=WhisperProcessor.from_pretrained(folder,local_files_only=True)
        processor.tokenizer.set_prefix_tokens(language='vi',task='transcribe',predict_timestamps=False)
        model=WhisperForConditionalGeneration.from_pretrained(folder,local_files_only=True,use_safetensors=False,
            dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        model.generation_config.forced_decoder_ids=None
        model.generation_config.language='vi';model.generation_config.task='transcribe'
        if checkpoint:model=PeftModel.from_pretrained(model,checkpoint,is_trainable=True)
        else:model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=cfg['dropout'],
            target_modules=cfg['targets'],bias='none'))
        clips=read(data/'train_asr_clips.json')
        if smoke:clips=sorted(clips,key=lambda r:len(words(r['text'])),reverse=True)[:2]
        train_rows=asr_dataset(clips,processor,torch,np)
        collator=asr_collator(processor,model.config.decoder_start_token_id,torch)
        eval_rows=None
    else:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig,Qwen3_5ForConditionalGeneration
        from efficient_attention import install,probe
        # Existing tested kernel wrapper expands GQA explicitly and forbids quadratic math fallback.
        probe();install()
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_snapshot(),local_files_only=True,
            quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        # Keep large frozen embeddings/vision BF16; norms and recurrent scale parameters stay FP32.
        for name,p in model.named_parameters():
            if not p.requires_grad and p.is_floating_point() and not isinstance(p,bnb.nn.Params4bit) and not any(x in name for x in ('norm','A_log','dt_bias')):
                p.data=p.data.to(torch.bfloat16)
        targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'language_model' in n
                 and not any(x in n for x in ('lm_head','embed_tokens','visual'))]
        assert targets and any('linear_attn' in n for n in targets)
        if checkpoint:model=PeftModel.from_pretrained(model,checkpoint,is_trainable=True)
        else:model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=cfg['dropout'],
            target_modules=targets,bias='none'))
        model.peft_config['default'].base_model_name_or_path=cfg['base'];model.peft_config['default'].revision=cfg['revision']
        processor=AutoTokenizer.from_pretrained(model_snapshot(),local_files_only=True)
        prof=read(output/'TOKEN_PROFILE.json')
        assert sha(output/'train_tokens.jsonl')==prof['partitions']['train']['token_cache_sha256']
        train_rows=[json.loads(l) for l in (output/'train_tokens.jsonl').read_text(encoding='utf8').splitlines()]
        if smoke:train_rows=sorted(train_rows,key=lambda r:r['length'],reverse=True)[:2]
        eval_rows=None if smoke else [json.loads(l) for l in (output/'val_tokens.jsonl').read_text(encoding='utf8').splitlines()]
        collator=lambda rows:collate_rows(rows,processor.pad_token_id,torch)
    model.config.use_cache=False
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and 'visual' not in n for n,p in trainable)
    params=sum(p.numel() for n,p in trainable)
    initial_b={n:p.detach().cpu().clone() for n,p in trainable if '.lora_B.' in n}
    args=dict(output_dir=str(run/'checkpoints'),per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else cfg['gradient_accumulation'],num_train_epochs=cfg['epochs'],
        learning_rate=cfg['learning_rate'],warmup_steps=.05,weight_decay=.01,lr_scheduler_type='linear',
        bf16=True,fp16=False,optim='adamw_torch',gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},logging_steps=1,logging_nan_inf_filter=False,
        torch_empty_cache_steps=1 if kind=='llm' else None,
        save_steps=1 if smoke else (50 if kind=='asr' else 10),save_strategy='steps',save_total_limit=2,
        eval_strategy='no',remove_unused_columns=False,dataloader_num_workers=0,report_to='none',
        seed=20261103,data_seed=20261103,max_grad_norm=1.0,push_to_hub=False,prediction_loss_only=True)
    if smoke:args['max_steps']=2
    training_args=TrainingArguments(**args)
    started=time.monotonic();val_history=read(run/'VAL_HISTORY.json') if resume and (run/'VAL_HISTORY.json').exists() else []
    class Progress(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            atomic(run/'PROGRESS.json',{'step':state.global_step,'total_steps':state.max_steps,'elapsed_s':time.monotonic()-started,
                'cuda_max_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_max_reserved_bytes':torch.cuda.max_memory_reserved()})
            if state.global_step>=state.max_steps:control.should_save=True
            return control
        def on_save(self,args,state,control,**kwargs):
            if kind=='asr' and not smoke and (state.global_step%250==0 or state.global_step>=state.max_steps):
                result=evaluate_asr(model,processor,read(data/'val_asr_clips.json'),torch,np,run,state.global_step)
                val_history.append({'step':state.global_step,'wer_percent':result['wer_percent'],'checkpoint':str(run/'checkpoints'/f'checkpoint-{state.global_step}')})
                best=min(val_history,key=lambda x:x['wer_percent'])
                # Save a separate best adapter before checkpoint retention removes its optimizer folder.
                if best['step']==state.global_step:
                    model.save_pretrained(run/'best_adapter',safe_serialization=True)
                    processor.save_pretrained(run/'processor')
                atomic(run/'VAL_HISTORY.json',val_history)
            return control
    if kind=='llm':
        from train_lora import completion_causal_loss
        class CompletionTrainer(Trainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                loss,outputs=completion_causal_loss(model,inputs,num_items_in_batch)
                return (loss,outputs) if return_outputs else loss
        trainer=CompletionTrainer(model=model,args=training_args,train_dataset=train_rows,eval_dataset=eval_rows,
            data_collator=collator,callbacks=[Progress()])
    else:
        trainer=Trainer(model=model,args=training_args,train_dataset=train_rows,data_collator=collator,callbacks=[Progress()])
        trainer.model_accepts_loss_kwargs=False
    atomic(run/'RUN_MANIFEST.json',{'status':'running','configuration':common,'trainable_parameters':params})
    print('TRAIN_START',kind,'smoke',smoke,'examples',len(train_rows),'trainable',params,'GPU',torch.cuda.get_device_name(0),flush=True)
    try:
        # Non-reentrant checkpointing retains layer inputs. Offload those saved tensors
        # to pinned host RAM rather than allowing WDDM to page the whole CUDA allocation.
        # Checkpoint recomputation, parameters, gradients and supervised tokens are unchanged.
        with torch.autograd.graph.save_on_cpu(pin_memory=True) if kind=='llm' else nullcontext():
            result=trainer.train(resume_from_checkpoint=checkpoint)
        final=run/'adapter_final';model.save_pretrained(final,safe_serialization=True);processor.save_pretrained(run/'processor')
        tensors=load_file(str(final/'adapter_model.safetensors'))
        assert all(torch.isfinite(t).all().item() for t in tensors.values())
        changed=any(not torch.equal(initial_b[n],p.detach().cpu()) for n,p in model.named_parameters() if n in initial_b)
        if checkpoint and not changed:
            changed=any(torch.count_nonzero(t).item()>0 for n,t in tensors.items() if '.lora_B.' in n)
        assert changed,'LoRA matrices did not change'
        val_loss=trainer.evaluate() if kind=='llm' and not smoke else None
        atomic(run/'RUN_MANIFEST.json',{'status':'completed','configuration':common,'trainable_parameters':params,
            'step':trainer.state.global_step,'total_steps':trainer.state.max_steps,'adapter':str(final),
            'adapter_changed':changed,'adapter_finite':True,'training_metrics':result.metrics,
            'val_loss':val_loss,'asr_val_history':val_history,'cuda_max_reserved_bytes':torch.cuda.max_memory_reserved()})
        print('TRAIN_COMPLETED',kind,smoke,final,flush=True)
    except BaseException as e:
        atomic(run/'RUN_MANIFEST.json',{'status':'failed_or_interrupted','configuration':common,'error':repr(e),
            'trainable_parameters':params,'cuda_max_reserved_bytes':torch.cuda.max_memory_reserved()})
        raise


def merge(data,output):
    verify(data,full=True)
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,WhisperForConditionalGeneration,AutoTokenizer,WhisperProcessor
    from peft import PeftModel
    for kind in ('asr','llm'):
        run=output/(kind+'_full');state=read(run/'RUN_MANIFEST.json');assert state['status']=='completed'
        adapter=run/'best_adapter' if kind=='asr' and (run/'best_adapter').exists() else run/'adapter_final'
        dest=output/'merged'/kind
        if dest.exists():raise RuntimeError('Preserve existing merged outputs; use a verified resume marker, do not overwrite')
        cfg=state['configuration']['config']
        if kind=='asr':
            base=HERE.parent/'gipformer_val_20261106/.runtime/models/medium'
            model=WhisperForConditionalGeneration.from_pretrained(base,local_files_only=True,use_safetensors=False,dtype=torch.float16,device_map={'':'cpu'})
            processor=WhisperProcessor.from_pretrained(run/'processor',local_files_only=True)
        else:
            model=Qwen3_5ForConditionalGeneration.from_pretrained(model_snapshot(),local_files_only=True,dtype=torch.bfloat16,device_map={'':'cpu'})
            processor=AutoTokenizer.from_pretrained(model_snapshot(),local_files_only=True)
        peft=PeftModel.from_pretrained(model,adapter,device_map={'':'cpu'})
        merged=peft.merge_and_unload(safe_merge=True);merged.config.use_cache=True
        merged.save_pretrained(dest,safe_serialization=True,max_shard_size='2GB');processor.save_pretrained(dest)
        json_new(dest/'MERGE_PROVENANCE.json',{'kind':kind,'base':cfg['base'],'revision':cfg['revision'],
            'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'parameters':sum(p.numel() for p in merged.parameters()),
            'adapter_merged':True,'source_manifest_sha256':sha(run/'RUN_MANIFEST.json')})
        del merged,peft,model;gc.collect()
    print('MERGE_COMPLETED',output/'merged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('profile','asr_smoke','llm_smoke','asr','llm','merge'),required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.data=a.data.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='profile':profile(a.data,a.output)
    elif a.stage=='merge':merge(a.data,a.output)
    else:training(a.data,a.output,a.stage.split('_')[0],a.stage.endswith('_smoke'),a.resume)
