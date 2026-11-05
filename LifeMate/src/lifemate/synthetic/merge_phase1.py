import json
from pathlib import Path
from .common import SYN_ROOT, MANIFEST, load_jsonl
recs=[]
for t in ["t1","t2","t3","t4c"]:
    f=SYN_ROOT/f"phase1_{t}.jsonl"
    if f.exists(): recs+=load_jsonl(f)
recs.sort(key=lambda r:r["id"])
(SYN_ROOT/"synthetic_phase1.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in recs)+"\n",encoding="utf-8")
by={}
for r in recs: by[r["technique"]]=by.get(r["technique"],0)+1
meta={"n":len(recs),"by_tech":by}
(SYN_ROOT/"synthetic_phase1_meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
(MANIFEST/"synth_phase1.json").write_text(json.dumps([{"id":r["id"],"technique":r["technique"],"source_id":r["source_id"]} for r in recs],ensure_ascii=False),encoding="utf-8")
print("merged:",meta)