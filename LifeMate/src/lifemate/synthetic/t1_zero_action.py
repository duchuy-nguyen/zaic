from copy import deepcopy
from .common import *
def hist_uses(s,ns):
    for t in s.get("history",[]):
        for c in t.get("tool_calls",[]) or []:
            if c["name"].split(".")[0] in ns: return True
    return False
def build_t1(b,samples,catalog,target=500):
    tmap,groups=tool_map(catalog),app_group(catalog)
    cands=[s for s in samples if s["label"]["tool_calls"] and not any(c["name"].startswith("sys.") for c in s["label"]["tool_calls"])]
    b.rng.shuffle(cands); made=0
    for s in cands:
        if made>=target: break
        ns=gold_ns(s); inst=s["context"]["installed_apps"]
        if not ns.issubset(inst): continue
        if hist_uses(s,ns): continue
        rem={groups[a] for a in inst if a not in ns}
        if rem & {groups[a] for a in ns}: continue
        b.assert_source(s["id"]); sid=b.new_id()
        ctx=deepcopy(s["context"])
        for a in ns: ctx["installed_apps"].pop(a,None)
        b.add({"id":sid,"source_id":s["id"],"technique":"t1_zero_action","audio":s["audio"],
               "audio_source":f"train:{s['id']}","transcript_mode":"gold","audio_text":s["audio_text"],
               "context":ctx,"history":s.get("history",[]),
               "label":{"tool_calls":[],"tool_results":[],"final_state":{}},
               "provenance":{"removed_apps":sorted(ns)}}); made+=1
    return made