import re
from .common import *
from .env_runner import make_env
ALIAS = {"amount":["amount","price","cost","value","quantity","qty","fee","money","tien","so_tien","budget","limit"],
         "temperature":["temperature","temp","degree","do","nhiet_do"],
         "time":["time","hour","minute","date","when","start","end","gio","ngay","thoi_gian","duration","deadline"]}
def _grp(p):
    pl=p.lower()
    for g,ks in ALIAS.items():
        if any(k in pl for k in ks): return g
    return None
ID_LIKE={"device_id","contact_id","account_id","reminder_id","playlist_id"}
RULE=[("amount",r"\b[\d\.]+\s*(nghìn|ngàn|k|triệu|tr)\b",""),
      ("temperature",r"\b[\d]+\s*(độ|degrees?)\b",""),
      ("time",r"\b(lúc|vào)\s+[\d]+\s*(giờ|h)\s*(rưỡi|sáng|chiều|tối)?\b","")]
PROMPT='Viet lai cau tieng Viet sau sao cho nguoi nghe KHONG suy ra duoc tham so "{p}" ({d}), giu nguyen thong tin khac, van phong noi tu nhien. Chi tra ve cau viet lai.\n cau goc: {t}'
def pick_removable(call,sc):
    props=sc.get("parameters",{}).get("properties",{}); req=sc.get("parameters",{}).get("required",[])
    out=[]
    for p in req:
        if p in ID_LIKE or p not in call["arguments"]: continue
        sp=props.get(p,{})
        if sp.get("type") in("integer","number","string") and "enum" not in sp and "x-ref" not in sp: out.append(p)
    return out
def rewrite(text, param, desc, llm=None):
    if llm is not None:
        try:
            r = llm(PROMPT.format(p=param, d=desc, t=text)).strip()
            if r and r != text:
                return r
        except Exception:
            pass
    g = _grp(param)                 # <-- chỗ bản cũ thiếu: GOI _grp de map ten param -> nhom
    if g is None:
        return None
    for k, pat, rep in RULE:
        if k != g:
            continue
        new = re.sub(pat, rep, text, flags=re.I)
        if new != text:
            return re.sub(r"\s+", " ", new).strip()
    return None
def build_t3(b,samples,catalog,target=375,llm=None):
    tmap=tool_map(catalog)
    cands=[s for s in samples if not any(c["name"].startswith("sys.") for c in s["label"]["tool_calls"])]
    b.rng.shuffle(cands); made=0
    for s in cands:
        if made>=target: break
        calls=s["label"]["tool_calls"]
        w=next((c for c in reversed(calls) if tmap[c["name"]]["kind"]=="write"),None)
        if not w: continue
        rm=pick_removable(w,tmap[w["name"]])
        if not rm: continue
        param=b.rng.choice(rm); desc=tmap[w["name"]]["parameters"]["properties"][param].get("description",param)
        nt=rewrite(s["audio_text"],param,desc,llm)
        if not nt: continue
        b.assert_source(s["id"]); sid=b.new_id(); b.write_state(sid,load_state("train",s["id"]))
        env=make_env(sid); pre=calls[:calls.index(w)]; obs=[]; ex=[]; ok=True
        for rc in pre:
            r=env.call(rc["name"],rc["arguments"])
            if not r.get("ok"): ok=False; break
            obs.append(r); ex.append(rc)
        if not ok: continue
        ask={"name":"sys.ask_missing_info","arguments":{"tool_name":w["name"],"params":[param]}}
        ra=env.call(ask["name"],ask["arguments"])
        b.add({"id":sid,"source_id":s["id"],"technique":"t3_param_removal","audio":s["audio"],
               "audio_source":f"train:{s['id']}","transcript_mode":"llm_rewrite" if llm else "rule_rewrite",
               "audio_text":nt,"context":s["context"],"history":s.get("history",[]),
               "label":{"tool_calls":ex+[ask],"tool_results":obs+[ra],"final_state":{}},
               "provenance":{"removed_param":param,"target_tool":w["name"],"original_text":s["audio_text"]}}); made+=1
    return made