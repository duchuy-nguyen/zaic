"""Exact step-DPO gradients in bounded memory, with original SFT replay.

Reference LPs cached from the actual baseline50.9 adapter. Dropout is disabled.
For each preference, obtain current detached LPs, then recompute and backprop
chosen and rejected separately using the exact chain-rule coefficients. No
optimizer update occurs between those passes; no joint two-trajectory graph.
"""
import argparse,gc,json,math,os,random,time
from pathlib import Path
from common import HERE,PARENT,ADAPTER,read,sha,new,atomic,verify
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128,garbage_collection_threshold:0.8')

def data(name):return [json.loads(l) for l in (HERE/'data'/name).open(encoding='utf8')]
def approved_data():
    review=read(HERE/'PAIR_REVIEW.json');assert review['status']=='approved'
    assert review['preferences_sha256']==sha(HERE/'data/preferences.jsonl')
    ids=set(review['approved_ids']);all_pairs=data('preferences_tokens.jsonl');old=data('replay_tokens.jsonl')
    indices=[j for j,p in enumerate(all_pairs) if p['id'] in ids]
    assert len(indices)==len(ids)>=48 and len({all_pairs[j]['source_id'] for j in indices})>=32
    return [all_pairs[j] for j in indices],[r for j in indices for r in old[j*3:j*3+3]]

def math_test():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(77)
    x=torch.randn(7,dtype=torch.float64,requires_grad=True);y=x.detach().clone().requires_grad_()
    c=torch.randn(7,dtype=torch.float64);r=torch.randn(7,dtype=torch.float64);beta=.1;ref=.37
    pc=torch.dot(x,c);pr=torch.dot(x,r)
    loss=-F.logsigmoid(beta*(pc-pr-ref));expected=torch.autograd.grad(loss,x)[0]
    with torch.no_grad():weight=beta*torch.sigmoid(-beta*(torch.dot(y,c)-torch.dot(y,r)-ref))
    (-weight*torch.dot(y,c)).backward();(weight*torch.dot(y,r)).backward()
    torch.testing.assert_close(y.grad,expected,rtol=1e-12,atol=1e-12)
    assert weight>0 and torch.isfinite(loss)
    new(HERE/'DPO_MATH_TEST.json',{'passed':True,'two_pass_gradient_matches_standard_DPO':True,'max_gradient_error':float((y.grad-expected).abs().max()),'dropout_must_be_disabled':True})
    print('DPO_CHAIN_RULE_PARITY_PASSED',flush=True)

def load(adapter):
    import torch,numpy as np,bitsandbytes as bnb
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    from efficient_attention import probe,install
    torch.set_num_threads(4);seed=20261115;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32=False
    probe();install()
    snapshot=read(PARENT/'.runtime/MODELS.json')['snapshot']
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    m=Qwen3_5ForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,quantization_config=quant,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
    m=prepare_model_for_kbit_training(m,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    for n,p in m.named_parameters():
        if not p.requires_grad and p.is_floating_point() and not isinstance(p,bnb.nn.Params4bit) and not any(k in n for k in ('norm','A_log','dt_bias')):p.data=p.data.to(torch.bfloat16)
    m=PeftModel.from_pretrained(m,adapter,is_trainable=True);m.config.use_cache=False
    visual=m.get_base_model().model.visual
    assert not any(p.requires_grad for p in visual.parameters())
    m.get_base_model().model.visual=None;del visual;gc.collect();torch.cuda.empty_cache()
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    trained=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _,p in trained)==32464896 and all('lora_' in n and 'visual' not in n for n,p in trained)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    return m,tok,trained

def logp(model,row):
    import torch
    import torch.nn.functional as F
    labels=torch.tensor([row['labels']],device='cuda',dtype=torch.long)
    inputs=torch.tensor([row['input_ids']],device='cuda',dtype=torch.long)
    shifted=labels[:,1:];positions=torch.nonzero(shifted.ne(-100).any(dim=0),as_tuple=False).flatten()
    assert positions.numel()==row['target_tokens'] and 0<positions.numel()<len(row['input_ids'])<=16384
    outputs=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False,labels=None,logits_to_keep=positions)
    logits=outputs.logits
    assert logits.shape[1]==len(positions)
    targets=shifted.index_select(1,positions)
    lp=-F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='sum')
    return lp

def reference(resume):
    verify();math_test()
    import torch
    pairs,_=approved_data();out=HERE/'outputs/reference'
    out.mkdir(parents=True,exist_ok=True);records=out/'records';records.mkdir(exist_ok=True)
    if (out/'RUN_MANIFEST.json').exists() and read(out/'RUN_MANIFEST.json')['status']=='completed':return
    committed=sorted(records.glob('*.json'))
    if committed:
        assert resume,'Reference cache interrupted; explicit resume required'
        assert [p.stem for p in committed]==[f'{j:04d}' for j in range(len(committed))]
        assert [read(p)['id'] for p in committed]==[p['id'] for p in pairs[:len(committed)]]
    model,tok,trained=load(ADAPTER);model.eval()
    atomic(out/'RUN_MANIFEST.json',{'status':'running','total':len(pairs),'completed':0,'parent_adapter_sha256':sha(ADAPTER/'adapter_model.safetensors')})
    for j,pair in enumerate(pairs):
        if j<len(committed):continue
        if (HERE/'PAUSE_REQUESTED').exists():atomic(out/'RUN_MANIFEST.json',{'status':'paused_by_user','completed':j,'total':len(pairs)});return
        with torch.no_grad():c=float(logp(model,pair['chosen']));r=float(logp(model,pair['rejected']))
        assert math.isfinite(c) and math.isfinite(r)
        atomic(records/f'{j:04d}.json',{'id':pair['id'],'chosen_logp':c,'rejected_logp':r,'data_sha256':sha(HERE/'data/preferences_tokens.jsonl')})
        atomic(out/'RUN_MANIFEST.json',{'status':'running','total':len(pairs),'completed':j+1,'parent_adapter_sha256':sha(ADAPTER/'adapter_model.safetensors')})
        print(f'REFERENCE [{j+1}/{len(pairs)}] {pair["id"]}',flush=True)
    assert sha(ADAPTER/'adapter_model.safetensors')==read(HERE/'PRESERVATION.json')['parent_adapter_sha256']
    atomic(out/'RUN_MANIFEST.json',{'status':'completed','total':len(pairs),'completed':len(pairs),'parent_adapter_sha256':sha(ADAPTER/'adapter_model.safetensors')})

def save_checkpoint(model,optimizer,scheduler,step,run):
    import torch,numpy as np
    path=run/'checkpoints'/f'checkpoint-{step}';path.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(path,safe_serialization=True)
    torch.save(optimizer.state_dict(),path/'optimizer.pt');torch.save(scheduler.state_dict(),path/'scheduler.pt')
    torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},path/'rng_state.pth')
    atomic(path/'trainer_state.json',{'global_step':step,'sampler':'fixed one preference and3 replay rows per update'})
    files=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
    atomic(run/'LATEST_CHECKPOINT.json',{'step':step,'path':str(path),'files_sha256':{n:sha(path/n) for n in files}})

def train(smoke,resume):
    verify();math_test();recipe=read(HERE/'recipe.json')
    for p,h in read(HERE/'DATA.json')['files_sha256'].items():assert sha(HERE/p)==h,p
    import torch,numpy as np
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from activation_memory import saved_activation_context
    from optimizer_memory import OffloadedAdamW
    pairs,replay=approved_data()
    assert read(HERE/'outputs/reference/RUN_MANIFEST.json')['status']=='completed'
    refs=[read(p) for p in sorted((HERE/'outputs/reference/records').glob('*.json'))]
    assert [r['id'] for r in refs]==[r['id'] for r in pairs]
    assert read(HERE/'outputs/reference/RUN_MANIFEST.json')['parent_adapter_sha256']==read(HERE/'PRESERVATION.json')['parent_adapter_sha256']
    assert all(r['data_sha256']==sha(HERE/'data/preferences_tokens.jsonl') for r in refs)
    if smoke:
        j=max(range(len(pairs)),key=lambda j:max(pairs[j]['chosen']['length'],pairs[j]['rejected']['length']))
        pairs,refs=[pairs[j]],[refs[j]];replay=sorted(replay,key=lambda r:r['length'],reverse=True)[:3]
    steps=1 if smoke else len(pairs);assert len(pairs)==steps and len(replay)==steps*3
    run=HERE/'outputs'/('smoke' if smoke else 'decision_dpo');run.mkdir(parents=True,exist_ok=True)
    marker=run/'RUN_MANIFEST.json';cfg={'freeze_sha256':sha(HERE/'FREEZE.json'),'parent_adapter_sha256':read(HERE/'PRESERVATION.json')['parent_adapter_sha256'],
        'data_sha256':sha(HERE/'DATA.json'),'pair_review_sha256':sha(HERE/'PAIR_REVIEW.json'),'steps':steps,'smoke':smoke,'beta':recipe['beta'],'learning_rate':recipe['learning_rate'],'replay_weight':recipe['replay_weight']}
    checkpoint=None;start_step=0
    if marker.exists():
        old=read(marker);assert old['configuration']==cfg
        if old['status']=='completed':return
        assert resume,'Explicit resume required'
        latest=read(run/'LATEST_CHECKPOINT.json');checkpoint=Path(latest['path']);start_step=latest['step']
        for n,h in latest['files_sha256'].items():assert sha(checkpoint/n)==h
    atomic(marker,{'status':'preparing','configuration':cfg,'step':start_step})
    model,tok,trained=load(checkpoint or ADAPTER);model.train()
    optimizer=OffloadedAdamW([p for _,p in trained],lr=recipe['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
    warmup=0 if smoke else 2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda k:1. if smoke else k/max(1,warmup) if k<warmup else max(0.,(steps-k)/max(1,steps-warmup)))
    if checkpoint:
        optimizer.load_state_dict(torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False))
        scheduler.load_state_dict(torch.load(checkpoint/'scheduler.pt',map_location='cpu',weights_only=False))
        rng=torch.load(checkpoint/'rng_state.pth',map_location='cpu',weights_only=False)
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
    atomic(marker,{'status':'running','configuration':cfg,'step':start_step,'trainable_parameters':32464896})
    started=time.monotonic()
    try:
        for j in range(start_step,steps):
            pair=pairs[j];ref=refs[j];optimizer.zero_grad(set_to_none=True)
            # Same params, dropout disabled, same complete inputs on both passes.
            with torch.no_grad():pc=float(logp(model,pair['chosen']));pr=float(logp(model,pair['rejected']))
            z=recipe['beta']*((pc-pr)-(ref['chosen_logp']-ref['rejected_logp']))
            weight=recipe['beta']*float(torch.sigmoid(torch.tensor(-z,dtype=torch.float64)))
            dpo_loss=float(torch.nn.functional.softplus(torch.tensor(-z,dtype=torch.float64)))
            assert math.isfinite(dpo_loss) and math.isfinite(weight)
            with saved_activation_context():
                lp=logp(model,pair['chosen']);c_check=float(lp.detach());(-weight*lp).backward();del lp
                lp=logp(model,pair['rejected']);r_check=float(lp.detach());(weight*lp).backward();del lp
                assert abs(c_check-pc)<.03 and abs(r_check-pr)<.03,'Recomputed LPs differ; refuse inconsistent DPO gradient'
                sft_losses=[]
                for r in replay[j*3:j*3+3]:
                    ce=-logp(model,r)/r['target_tokens'];sft_losses.append(float(ce.detach()))
                    (recipe['replay_weight']/3*ce).backward();del ce
            grad=torch.nn.utils.clip_grad_norm_([p for _,p in trained],1.)
            assert math.isfinite(float(grad))
            optimizer.step();scheduler.step();step=j+1
            record={'step':step,'total_steps':steps,'pair_id':pair['id'],'dpo_loss':dpo_loss,'replay_ce_mean':sum(sft_losses)/3,
                'policy_margin':pc-pr,'reference_margin':ref['chosen_logp']-ref['rejected_logp'],'grad_norm':float(grad),
                'learning_rate_next_step':scheduler.get_last_lr()[0],'elapsed_s':time.monotonic()-started,
                'cuda_max_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_max_reserved_bytes':torch.cuda.max_memory_reserved()}
            atomic(run/'PROGRESS.json',record)
            with (run/'metrics.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
            if step==1 or step%10==0 or step==steps or (HERE/'PAUSE_REQUESTED').exists():save_checkpoint(model,optimizer,scheduler,step,run)
            print('DPO_STEP',json.dumps(record),flush=True)
            if (HERE/'PAUSE_REQUESTED').exists():atomic(marker,{'status':'paused_by_user','configuration':cfg,'step':step});return
            gc.collect();torch.cuda.empty_cache()
        final=run/'adapter_final';model.save_pretrained(final,safe_serialization=True);tok.save_pretrained(run/'tokenizer')
        saved=load_file(str(final/'adapter_model.safetensors'));state=get_peft_model_state_dict(model)
        assert set(saved)==set(state) and all(torch.isfinite(v).all().item() and torch.equal(v,state[n].detach().cpu()) for n,v in saved.items())
        assert sha(final/'adapter_model.safetensors')!=cfg['parent_adapter_sha256']
        new(run/'WEIGHTS_VERIFIED.json',{'finite':True,'changed_from_parent':True,'exact_readback':True,'adapter_sha256':sha(final/'adapter_model.safetensors')})
        atomic(marker,{'status':'completed','configuration':cfg,'step':steps,'total_steps':steps,'parent_steps':489,'new_phase_steps':steps,
            'adapter':str(final),'trainable_parameters':32464896,'metrics':{'elapsed_s':time.monotonic()-started}})
        print('DPO_TRAIN_COMPLETED',steps,flush=True)
    except BaseException as e:atomic(marker,{'status':'failed_or_interrupted','configuration':cfg,'step':j,'error':repr(e)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['math','reference','smoke','train']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    math_test() if a.stage=='math' else reference(a.resume) if a.stage=='reference' else train(a.stage=='smoke',a.resume)
