from __future__ import annotations
import json, re, unicodedata
from collections import Counter
from pathlib import Path
OFFICIAL = Path("official/lifemate-v3.2.0")
SYN_ROOT = Path("synthetic")
MANIFEST = Path("manifests")
RUNS = Path("runs")

def norm_text(s): return re.sub(r"\s+"," ",unicodedata.normalize("NFC",s or "").lower()).strip()
def tokens(s):    return re.findall(r"[0-9a-zà-ỹ]+",norm_text(s))
def token_f1(a,b):
    ta,tb=Counter(tokens(a)),Counter(tokens(b)); inter=sum((ta&tb).values())
    if not inter: return 0.0
    p,r=inter/sum(ta.values()),inter/sum(tb.values()); return 2*p*r/(p+r)
def load_jsonl(p): return [json.loads(l) for l in open(p,encoding="utf-8")]
def load_catalog(split="train"):
    return json.loads((OFFICIAL/split/f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
def tool_map(cat):  return {t["name"]:t for a in cat["apps"] for t in a.get("tools",[])}
def app_group(cat): return {a["id"]:a.get("group") for a in cat["apps"]}
def gold_ns(s):     return {c["name"].split(".")[0] for c in s["label"]["tool_calls"] if not c["name"].startswith("sys.")}
def xref_params(sc):
    return [k for k,v in sc.get("parameters",{}).get("properties",{}).items() if "x-ref" in v]
def load_state(split,sid):
    return json.loads((OFFICIAL/split/f"{split}_environment"/sid/"state.json").read_text(encoding="utf-8"))
def count_name_hits(o,name):
    if isinstance(o,dict):  return (1 if norm_text(o.get("name",""))==norm_text(name) else 0)+sum(count_name_hits(v,name) for v in o.values())
    if isinstance(o,list):  return sum(count_name_hits(v,name) for v in o)
    return 0
def collect_names(o,acc=None):
    acc=[] if acc is None else acc
    if isinstance(o,dict):
        if isinstance(o.get("name"),str): acc.append(o["name"])
        for v in o.values(): collect_names(v,acc)
    elif isinstance(o,list):
        for v in o: collect_names(v,acc)
    return acc
def audio_refs(samples):
    for s in samples:
        yield s["audio"]
        for t in s.get("history",[]):
            if t.get("role")=="user" and t.get("audio"): yield t["audio"]