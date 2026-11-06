from collections import Counter
from pathlib import Path
import json
from .common import (OFFICIAL, SYN_ROOT, load_jsonl, load_catalog, tool_map,
                     norm_text, load_state, token_f1, collect_names)
from .env_runner import make_env, replay

fit=set(json.loads(Path("manifests/train_fit_provisional.json").read_text()))
samples=[s for s in load_jsonl(OFFICIAL/"train"/"train.jsonl") if s["id"] in fit]
tmap=tool_map(load_catalog("train"))
conf=json.loads(Path("runs/asr_cache/confusion.json").read_text(encoding="utf-8"))

c=Counter(); ex=[]; n=0
for s in samples:
    if not s["label"]["tool_calls"]: c["skip_no_calls"]+=1; continue
    toks=norm_text(s["audio_text"]).split()
    hit=[i for i,t in enumerate(toks) if t in conf and conf[t]]
    if not hit: c["A_no_conf_token"]+=1; continue
    c["pass_A"]+=1
    read=next((x for x in s["label"]["tool_calls"] if tmap[x["name"]]["kind"]=="read"),None)
    if not read: c["B_no_read"]+=1; continue
    n+=1; sid=f"diagt4_{n:05d}"
    d=SYN_ROOT/"synthetic_environment"/sid; d.mkdir(parents=True,exist_ok=True)
    (d/"state.json").write_text(json.dumps(load_state("train",s["id"]),ensure_ascii=False),encoding="utf-8")
    env=make_env(sid)
    obs=env.call(read["name"],read["arguments"])
    if not obs.get("ok"): c["C_read_fail"]+=1; continue
    names=collect_names(obs)
    if not names: c["D_no_names"]+=1; continue
    i=hit[0]; noisy_tok=list(conf[toks[i]])[0]
    sc=sorted((token_f1(noisy_tok,x),x) for x in names); best=sc[-1][0]
    if best<0.6:
        c["E_f1_below_0.6"]+=1
        if len(ex)<15: ex.append((toks[i],noisy_tok,round(best,3),names[:3]))
        continue
    if len(sc)>1 and sc[-1][0]==sc[-2][0]: c["F_tie"]+=1; continue
    if replay(env,s["label"]["tool_calls"]) is None: c["G_replay_fail"]+=1; continue
    c["ACCEPT"]+=1
print("FUNNEL_T4:",dict(c))
print("--- E-examples (gold_tok, noisy_tok, best_f1, names) ---")
for e in ex: print(e)