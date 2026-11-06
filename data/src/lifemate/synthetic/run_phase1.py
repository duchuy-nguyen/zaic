import argparse, json
from pathlib import Path
from .common import OFFICIAL, RUNS, load_jsonl, load_catalog
from .builder import SyntheticBuilder
def load_conf(): return json.loads((RUNS/"asr_cache"/"confusion.json").read_text(encoding="utf-8"))
def load_llm():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        mdl=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",torch_dtype=torch.float16,device_map="auto")
        def f(prompt):
            ids=tok(prompt,return_tensors="pt").to(mdl.device)
            return tok.decode(mdl.generate(**ids,max_new_tokens=128,do_sample=False)[0],skip_special_tokens=True)
        return f
    except Exception as e:
        print("[warn] LLM rewrite khong san sang, fallback rule:",e); return None
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tech",required=True,choices=["t1","t2","t3","t4c"])
    ap.add_argument("--target",type=int,default=0)
    ap.add_argument("--mode",choices=["rule","llm"],default="rule")
    a=ap.parse_args()
    fit=set(json.loads(Path("manifests/train_fit_provisional.json").read_text()))
    samples=[s for s in load_jsonl(OFFICIAL/"train"/"train.jsonl") if s["id"] in fit]
    cat=load_catalog("train"); b=SyntheticBuilder(a.tech,fit)
    if   a.tech=="t1": from .t1_zero_action import build_t1; n=build_t1(b,samples,cat,a.target)
    elif a.tech=="t2": from .t2_state_mutation import build_t2; n=build_t2(b,samples,cat,a.target)
    elif a.tech=="t3": from .t3_param_removal import build_t3; n=build_t3(b,samples,cat,a.target,llm=(load_llm() if a.mode=="llm" else None))
    else: from .t4_asr_noise import build_t4; n=build_t4(b,samples,cat,load_conf(),a.target)
    print(b.finalize(),"accepted:",n)
if __name__=="__main__": main()