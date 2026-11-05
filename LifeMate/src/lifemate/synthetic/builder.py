import json, shutil, random
from pathlib import Path
from .common import OFFICIAL, SYN_ROOT, load_state
class SyntheticBuilder:
    def __init__(self,tech,fit_ids,seed=20261103):
        self.tech,self.fit_ids,self.rng=tech,fit_ids,random.Random(seed)
        self.records,self.n=[],0
        SYN_ROOT.mkdir(parents=True,exist_ok=True)
        (SYN_ROOT/"synthetic_environment").mkdir(exist_ok=True)
        if not (SYN_ROOT/"zaic_tools").exists():
            shutil.copytree(OFFICIAL/"train"/"zaic_tools",SYN_ROOT/"zaic_tools")
        if not (SYN_ROOT/"synthetic_tools_catalog.json").exists():
            shutil.copy2(OFFICIAL/"train"/"train_tools_catalog.json",SYN_ROOT/"synthetic_tools_catalog.json")
    def assert_source(self,sid): assert sid in self.fit_ids, f"LEAK RISK: {sid} ngoai TRAIN_FIT"
    def new_id(self): self.n+=1; return f"synth_{self.tech}_{self.n:05d}"
    def write_state(self,syn_id,state):
        d=SYN_ROOT/"synthetic_environment"/syn_id; d.mkdir(parents=True,exist_ok=True)
        (d/"state.json").write_text(json.dumps(state,ensure_ascii=False),encoding="utf-8")
    def add(self,rec): self.records.append(rec)
    def finalize(self):
        self.records.sort(key=lambda r:r["id"])
        out=SYN_ROOT/f"phase1_{self.tech}.jsonl"
        out.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in self.records)+"\n",encoding="utf-8")
        return {"file":str(out),"n":len(self.records)}