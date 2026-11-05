import json, hashlib
from pathlib import Path
from .common import OFFICIAL, MANIFEST
rows=[json.loads(l) for l in open(OFFICIAL/"train"/"train.jsonl",encoding="utf-8")]
def gkey(s):
    h=json.dumps(s.get("history",[]),sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(h.encode()).hexdigest()
groups=sorted({gkey(s) for s in rows})
val=set(groups[:int(len(groups)*0.2)])
fit=[s["id"] for s in rows if gkey(s) not in val]
MANIFEST.mkdir(parents=True,exist_ok=True)
(MANIFEST/"train_fit_provisional.json").write_text(json.dumps(fit),encoding="utf-8")
print("TRAIN_FIT tam:",len(fit),"/ VAL:",len(rows)-len(fit))