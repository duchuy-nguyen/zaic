"""Actual tiny Qwen/BNB/PEFT/Trainer/AMP/save/resume check, no model download.

Uses randomly initialized tiny weights and synthetic token IDs. Long attention
probe uses real Qwen4B attention head geometry, not real model parameter count.
Run with the pinned training stack; outputs contain no BTC data.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'local_baseline')]
import worker
from core import write_json, append_jsonl


def run(directory, long_sequence=0):
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, PreTrainedTokenizerFast, TrainerCallback
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from peft import get_peft_model_state_dict
    if not torch.cuda.is_available(): raise RuntimeError('CUDA needed for actual NF4/AMP check')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    base = directory / 'tiny_base'; base.mkdir(exist_ok=True)
    torch.manual_seed(7); torch.set_num_threads(2)
    config = Qwen3_5Config(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=16,num_key_value_heads=4,head_dim=256,
        linear_num_value_heads=4,linear_num_key_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,
        linear_conv_kernel_dim=4,layer_types=['linear_attention','full_attention'],
        max_position_embeddings=16384,bos_token_id=3,eos_token_id=1,pad_token_id=1),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,
            spatial_merge_size=1,temporal_patch_size=1,out_hidden_size=64,deepstack_visual_indexes=[]))
    Qwen3_5ForConditionalGeneration(config).save_pretrained(base)
    vocab = {'[UNK]':0,'<|endoftext|>':1,'<|im_end|>':2,'<|im_start|>':3,'fixture':4,'reply':5}
    vocab.update({'t'+str(i):i for i in range(6,512)})
    raw = Tokenizer(WordLevel(vocab=vocab,unk_token='[UNK]')); raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',
        eos_token='<|im_end|>',pad_token='<|endoftext|>')
    tokenizer.chat_template = "{% for m in messages %}{{'<|im_start|>'+m.role+' '+m.content+'<|im_end|> '}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant '}}{% endif %}"
    tokenizer.save_pretrained(base)
    # Check the custom HF mask interface and cached generation against native
    # attention before testing quantization. No pretrained weights are used.
    from bounded_attention import register
    backend=register(3)
    native=Qwen3_5ForConditionalGeneration.from_pretrained(base,attn_implementation='sdpa',dtype=torch.float32).cpu().eval()
    bounded=Qwen3_5ForConditionalGeneration.from_pretrained(base,attn_implementation=backend,dtype=torch.float32).cpu().eval()
    sample=torch.randint(6,512,(1,17));mask=torch.ones_like(sample)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel([SDPBackend.MATH]):
        expected=native(input_ids=sample,attention_mask=mask,use_cache=False).logits
        observed=bounded(input_ids=sample,attention_mask=mask,use_cache=False).logits
        torch.testing.assert_close(expected,observed,atol=2e-5,rtol=2e-5)
        expected.square().mean().backward();observed.square().mean().backward()
        gradient_difference=0.
        for a,b in zip(native.parameters(),bounded.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=2e-4)
                gradient_difference=max(gradient_difference,(a.grad-b.grad).abs().max().item())
        with torch.inference_mode():
            options=dict(input_ids=sample,attention_mask=mask,max_new_tokens=3,min_new_tokens=3,
                         do_sample=False,eos_token_id=[1,2],pad_token_id=1)
            a=native.generate(**options)
            b=bounded.generate(**options,prefill_chunk_size=5)
        if not torch.equal(a,b):raise AssertionError('Cached generation differs between attention backends')
    forward_difference=(expected-observed).abs().max().item()
    del native,bounded,expected,observed,a,b;gc.collect()
    template = dict(mode='pilot',seed=7,deterministic=True,data_root=str(directory),
        student_id=str(base),student_revision='local-fixture',student_lora_r=2,
        student_train_context=8192,inference_context=16384,max_new_tokens=4,
        attention_query_chunk=128,student_loss_chunk=8,
        minimum_training_coverage=1.,student_grad_accum=1,student_lr=1e-4,student_epochs=2,
        limits={'student_max_steps':2},save_steps=1)
    def cfg_for(name):
        out=directory/name; out.mkdir(exist_ok=True)
        for _ in range(2):
            append_jsonl(out/'student_examples.jsonl',{'source_id':'fixture','sources':['fixture'],
                'messages':[{'role':'user','content':'fixture'}],'tools':[],
                'target':{'role':'assistant','content':'reply'}})
        return {**template,'run_dir':str(out)}
    baseline=cfg_for('baseline'); worker.student_train(baseline)
    resumed=cfg_for('resumed')
    class Pause(TrainerCallback):
        stopped=False
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step==1:
                self.stopped=True;control.should_training_stop=True;control.should_save=True
            return control
    try:
        with patch.object(worker,'deadline_callback',return_value=Pause()): worker.student_train(resumed)
    except worker.SessionBudget:
        pass
    else: raise AssertionError('Expected controlled pause after real optimizer step 1')
    if not worker.last_checkpoint(Path(resumed['run_dir'])/'student_checkpoints'):
        raise AssertionError('Resumable adapter/optimizer checkpoint missing')
    gc.collect();torch.cuda.empty_cache()
    worker.student_train(resumed)
    states=[]
    for cfg in (baseline,resumed):
        model,_=worker.load_llm(cfg,adapter=Path(cfg['run_dir'])/'student_adapter')
        states.append({k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()})
        del model;gc.collect();torch.cuda.empty_cache()
    if states[0].keys()!=states[1].keys():raise AssertionError('Adapter keys changed on resume')
    maximum=max((states[0][k]-states[1][k]).abs().max().item() for k in states[0])
    if maximum>1e-6:raise AssertionError('Resume adapter differs from uninterrupted run: '+str(maximum))
    report={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
            'python':sys.version,'cuda':torch.version.cuda,
            'compute_capability':list(torch.cuda.get_device_capability(0)),
            'packages':{name:worker.version(name) for name in ('transformers','peft','accelerate','bitsandbytes','tokenizers','huggingface-hub')},
            'optimizer_steps':2,'adapter_reload':True,'checkpoint_resume':True,
            'resume_max_adapter_difference':maximum,'full_4b_weights_tested':False,
            'native_vs_bounded_max_logit_difference':forward_difference,
            'native_vs_bounded_max_gradient_difference':gradient_difference,
            'cached_generation_matches_native':True}
    if long_sequence:
        model,_=worker.load_llm(baseline,train=True)
        ids=torch.randint(6,512,(1,long_sequence),device='cuda:0')
        labels=torch.full_like(ids,-100);labels[:,-32:]=ids[:,-32:]
        result=worker.student_smoke_steps(model,{'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels},baseline)
        report['long_attention_probe']={**result,'head_geometry':'16 Q heads, 4 KV heads, head_dim=256; tiny backbone'}
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(directory/'report.json',report)
    print('ACTUAL STACK CHECK PASSED:',json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--long-sequence',type=int,default=0);args=parser.parse_args()
    run(args.output,args.long_sequence)
