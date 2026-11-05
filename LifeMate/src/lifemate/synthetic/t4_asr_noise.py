from .common import *
from .env_runner import make_env, replay
def build_t4(b,samples,catalog,conf,target=250):
    tmap=tool_map(catalog)
    cands=[s for s in samples if s["label"]["tool_calls"]]
    b.rng.shuffle(cands); made=0
    for s in cands:
        if made>=target: break
        toks=norm_text(s["audio_text"]).split()
        hit=[i for i,t in enumerate(toks) if t in conf and conf[t]]
        if not hit: continue
        i=b.rng.choice(hit); noisy_tok=b.rng.choice(list(conf[toks[i]]))
        noisy=" ".join(toks[:i]+[noisy_tok]+toks[i+1:])
        read=next((c for c in s["label"]["tool_calls"] if tmap[c["name"]]["kind"]=="read"),None)
        if not read: continue
        b.assert_source(s["id"]); sid=b.new_id(); b.write_state(sid,load_state("train",s["id"]))
        env=make_env(sid); obs=env.call(read["name"],read["arguments"])
        if not obs.get("ok"): continue
        names=collect_names(obs); sc=sorted((token_f1(noisy_tok,n),n) for n in names)
        if not sc or sc[-1][0]<0.6: continue
        if len(sc)>1 and sc[-1][0]==sc[-2][0]: continue
        full=replay(env,s["label"]["tool_calls"])
        if full is None: continue
        b.add({"id":sid,"source_id":s["id"],"technique":"t4_asr_noise","audio":s["audio"],
               "audio_source":f"train:{s['id']}","transcript_mode":"asr_noisy_sim","audio_text":noisy,
               "context":s["context"],"history":s.get("history",[]),
               "label":{"tool_calls":s["label"]["tool_calls"],"tool_results":full,"final_state":s["label"]["final_state"]},
               "provenance":{"noise_ops":[{"pos":i,"gold":toks[i],"noisy":noisy_tok}],"conf_source":"phowhisper-small/train_fit"}}); made+=1
    return made