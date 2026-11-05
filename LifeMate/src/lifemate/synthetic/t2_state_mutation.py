from copy import deepcopy
from .common import *
from .mutators import duplicate_entity, groundable
from .env_runner import make_env
def build_t2(b,samples,catalog,target=625):
    tmap=tool_map(catalog)
    cands=[s for s in samples if not any(c["name"].startswith("sys.") for c in s["label"]["tool_calls"])]
    b.rng.shuffle(cands); made=0
    for s in cands:
        if made>=target: break
        calls=s["label"]["tool_calls"]
        reads=[c for c in calls if tmap[c["name"]]["kind"]=="read"]
        writes=[c for c in calls if tmap[c["name"]]["kind"]=="write"]
        if not reads or not writes: continue
        w=writes[0]; app=w["name"].split(".")[0]; refs=xref_params(tmap[w["name"]])
        if not refs: continue
        rp=refs[0]; rid=w["arguments"].get(rp)
        if not rid: continue
        state = load_state("train", s["id"])
        grounds = duplicate_entity(state, rid, b.rng)          # generic, không cần app
        if not grounds or not groundable(grounds, s["audio_text"]): continue
        b.assert_source(s["id"]); sid = b.new_id(); b.write_state(sid, state)
        env = make_env(sid); obs = []; ex = []; ok = True
        for rc in reads:
            r = env.call(rc["name"], rc["arguments"])
            if not r.get("ok"): ok = False; break
            obs.append(r); ex.append(rc)
        if not ok: continue
        ambig = any(count_name_hits(obs, g) >= 2 for g in grounds)   # read THẬT phải trả trùng
        if not ambig: continue
        ask={"name":"sys.ask_missing_info","arguments":{"tool_name":w["name"],"params":[rp]}}
        ra=env.call(ask["name"],ask["arguments"])
        b.add({"id":sid,"source_id":s["id"],"technique":"t2_state_mutation","audio":s["audio"],
               "audio_source":f"train:{s['id']}","transcript_mode":"gold","audio_text":s["audio_text"],
               "context":s["context"],"history":s.get("history",[]),
               "label":{"tool_calls":ex+[ask],"tool_results":obs+[ra],"final_state":{}},
               "provenance":{"mutated_app":app,"grounds":grounds,"duplicated_id":str(rid)+"_syn"}}); made+=1
    return made