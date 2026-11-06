import json, difflib
from collections import Counter, defaultdict
from pathlib import Path
from .common import OFFICIAL, RUNS, norm_text, load_jsonl
fit=set(json.loads(Path("manifests/train_fit_provisional.json").read_text()))
samples=[s for s in load_jsonl(OFFICIAL/"train"/"train.jsonl") if s["id"] in fit]
gold={}
for s in samples:
    gold[s["audio"]]=s["audio_text"]
    for t in s.get("history",[]):
        if t.get("role")=="user" and t.get("audio"): gold[t["audio"]]=t.get("audio_text","")
pred={json.loads(l)["audio"]:json.loads(l)["text"] for l in (RUNS/"asr_cache"/"transcripts.jsonl").open(encoding="utf-8")}
conf=defaultdict(Counter)
for a,g in gold.items():
    if a not in pred: continue
    gt,pt=norm_text(g).split(),norm_text(pred[a]).split()
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,gt,pt).get_opcodes():
        if tag=="replace" and (i2-i1)==(j2-j1):
            for x,y in zip(gt[i1:i2],pt[j1:j2]):
                if x!=y: conf[x][y]+=1
out={a:dict(c.most_common(5)) for a,c in conf.items() if sum(c.values())>=2}
(RUNS/"asr_cache"/"confusion.json").write_text(json.dumps(out,ensure_ascii=False),encoding="utf-8")
print("confusion entries:",len(out))