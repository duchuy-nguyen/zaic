import json, argparse, random
from pathlib import Path
from .common import OFFICIAL, SYN_ROOT, norm_text, load_jsonl
from .env_runner import make_env
ap=argparse.ArgumentParser(); ap.add_argument("--smoke",type=int,default=50); A=ap.parse_args()
fit=set(json.loads(Path("manifests/train_fit_provisional.json").read_text()))
allrows=load_jsonl(OFFICIAL/"train"/"train.jsonl")
val_text={norm_text(s["audio_text"]) for s in allrows if s["id"] not in fit}
recs=[]
for t in ["t1","t2","t3","t4c"]:
    f=SYN_ROOT/f"phase1_{t}.jsonl"
    if f.exists(): recs+=load_jsonl(f)
CHK={"t1_zero_action":lambda r:r["label"]["tool_calls"]==[] and r["label"]["final_state"]=={},
     "t2_state_mutation":lambda r:r["label"]["tool_calls"][-1]["name"]=="sys.ask_missing_info" and r["label"]["final_state"]=={},
     "t3_param_removal":lambda r:r["label"]["tool_calls"][-1]["name"]=="sys.ask_missing_info",
     "t4_asr_noise":lambda r:any(op.get("gold")!=op.get("noisy") for op in r["provenance"].get("noise_ops",[]))}
errs=[]; seen=set()
for r in recs:
    if not CHK[r["technique"]](r): errs.append(("structural",r["id"],r["technique"]))
    if norm_text(r["audio_text"]) in val_text: errs.append(("leak_val",r["id"],r["technique"]))
    h=hash((norm_text(r["audio_text"]),json.dumps(r["context"],sort_keys=True,ensure_ascii=False)))
    if h in seen: errs.append(("dup",r["id"],r["technique"]))
    seen.add(h)
print("structural/leak/dup errors:",len(errs))
for e in errs: print("  ERR",e)
pool=[r for r in recs if r["label"]["tool_calls"]]
random.seed(0); smoke=random.sample(pool,min(A.smoke,len(pool))); fail=0
print("smoke pool (records co label):",len(pool))
for r in smoke:
    env=make_env(r["id"])
    for c in r["label"]["tool_calls"]:
        v=env.validate(c["name"],c["arguments"])
        if not v.get("ok"): fail+=1; print("schema fail",r["id"],c["name"],v.get("error")); break
        ex=env.call(c["name"],c["arguments"])
        if not ex.get("ok"): fail+=1; print("exec fail",r["id"],c["name"]); break
print("smoke failures:",fail,"/",len(smoke))
print("VALIDATE DONE" if not errs and fail==0 else "VALIDATE HAS ERRORS")