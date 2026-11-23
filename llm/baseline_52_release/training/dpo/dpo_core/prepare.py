"""Original DPO preference construction functions, portable inputs supplied by wrapper."""

import argparse,collections,copy,json,random,shutil,sys,time
from pathlib import Path
from datetime import datetime,timedelta
from common import HERE,ROOT,CONTROL,PARENT,ADAPTER,read,sha,new,atomic,EXCLUDED_SOURCES


def rows(p):return [json.loads(l) for l in p.open(encoding='utf8')]

def write_rows(p,rs):
    with p.open('x',encoding='utf8') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def callmsg(c):return {'role':'assistant','content':'','tool_calls':[{'id':'call_next','type':'function','function':copy.deepcopy(c)}]}

FINISH={'role':'assistant','content':''}

def harvest(resume):
    from baseline50 import actor
    from baseline50.backend import LocalServer
    from baseline50.io import acquire_lock
    source=read(HERE/'SOURCE.json')
    for p,h in source['external_sha256'].items():assert sha(p)==h,p
    assert sha(source['baseline_model'])==source['baseline_model_sha256']
    plan=read(HERE/'inputs/plan.json');raw={r['id']:r for r in rows(HERE/'inputs/train_sources.jsonl')}
    sys.path.insert(0,source['training_sdk'])
    from zaic_tools import ToolEnv
    from zaic_tools.executor import Simulator
    from zaic_tools.utils import state_hash
    cat=read(source['catalog']);tools={t['name']:t for a in cat['apps'] for t in a['tools']}
    dataset=Path(source['training_sdk']).parent
    actor.bind_package(CONTROL/'assets_large/private_data')
    lock=acquire_lock(HERE/'harvest/run.lock');directory=HERE/'harvest/records';directory.mkdir(exist_ok=True)
    committed={p.stem:read(p) for p in directory.glob('*.json')}
    assert set(committed)=={s['id'] for s in plan[:len(committed)]}
    inflight=HERE/'harvest/inflight.json'
    if inflight.exists():assert read(inflight)['id'] in committed,'Uncommitted TRAIN rollout; investigate, do not auto rerun';inflight.unlink()
    marker=HERE/'harvest/RUN_MANIFEST.json'
    if marker.exists():assert resume
    state={'status':'running','completed_sources':len(committed),'total_sources':len(plan)};atomic(marker,state)
    class Capture:
        def __init__(self,backend):self.backend=backend;self.inputs=[]
        def generate(self,messages,tools,deadline):
            self.inputs.append({'prompt':copy.deepcopy(messages),'tools':copy.deepcopy(tools)})
            return self.backend.generate(messages,tools,deadline)
    try:
        session=len(list((HERE/'harvest').glob('server-*')))+1
        with LocalServer(source['baseline_model'],CONTROL/'assets_large/runtime/llama-server.exe',HERE/'harvest'/f'server-{session}') as backend:
            for spec in plan[len(committed):]:
                if (HERE/'PAUSE_REQUESTED').exists():state['status']='paused_by_user';atomic(marker,state);return
                sid=spec['id'];row=raw[sid]
                # A read-only TRAIN audit, distinct from the one agent environment.
                states=read(Path(source['training_sdk'])/f'train_environment/{sid}/state.json')
                ctx=copy.deepcopy(row['context'])
                for app,a in ctx['installed_apps'].items():a['state']=copy.deepcopy(states.get(app,{}))
                sim=Simulator({'context':ctx},cat)
                gold_results=[sim.call(c['name'],c['arguments']) for c in row['label']['tool_calls']]
                gold_ok=gold_results==row['label']['tool_results'] and all(o.get('ok') and o.get('done') is not False for o in gold_results)
                view={'id':sid,'latest_text':row['audio_text'],'context':row['context'],
                    'history':[({'role':'user','text':t['audio_text']} if t['role']=='user' else {k:t[k] for k in ('role','text','tool_calls','tool_results') if k in t}) for t in row['history']]}
                atomic(inflight,{'id':sid,'phase':'before_agent'})
                env=ToolEnv('train',sid,root=dataset);capture=Capture(backend)
                result=actor.infer(view,tools,env,capture,seconds=30,max_calls=8,max_repairs=2)
                item={**spec,'result':result,'step_inputs':capture.inputs,'gold_replay_ok':gold_ok,
                      'gold_final_state_hash':state_hash(sim.states),'gold_calls':row['label']['tool_calls']}
                atomic(directory/f'{sid}.json',item);committed[sid]=item;inflight.unlink()
                state.update(completed_sources=len(committed),last_id=sid,last_status=result['status']);atomic(marker,state)
                print(f"[TRAIN {len(committed)}/{len(plan)}] {sid} {spec['stratum']} {result['status']} {result['elapsed_s']:.2f}s",flush=True)
                if result['status'] in ('backend_error','execution_error') or backend.process.poll() is not None:raise RuntimeError('Infrastructure error committed, stop without selective retry')
        state.update(status='completed',completed_sources=len(committed));atomic(marker,state)
    except BaseException as e:state.update(status='failed',error=repr(e));atomic(marker,state);raise
    finally:lock.close()

def build():
    source=read(HERE/'SOURCE.json');assert read(HERE/'harvest/RUN_MANIFEST.json')['status']=='completed'
    sys.path.insert(0,source['training_sdk'])
    from zaic_tools.executor import Simulator
    from zaic_tools.utils import state_hash
    cat=read(source['catalog']);raw={r['id']:r for r in rows(HERE/'inputs/train_sources.jsonl')}
    split=read(source['split']);pool=[];exclusions=collections.Counter()
    for spec in read(HERE/'inputs/plan.json'):
        sid=spec['id'];record=read(HERE/'harvest/records'/f'{sid}.json');row=raw[sid];result=record['result'];gold=row['label']['tool_calls']
        if sid in set(EXCLUDED_SOURCES):
            exclusions['unresolved_prior_TRAIN_annotation_audit']+=1;continue
        if not record['gold_replay_ok'] or result['repairs'] or result['status'] not in ('finished','asked_missing_info','call_budget_exhausted'):
            exclusions['source_unverified_repaired_or_incomplete']+=1;continue
        states=read(Path(source['training_sdk'])/f'train_environment/{sid}/state.json');ctx=copy.deepcopy(row['context'])
        for app,a in ctx['installed_apps'].items():a['state']=copy.deepcopy(states.get(app,{}))
        sim=Simulator({'context':ctx},cat)
        for j,(entry,inp) in enumerate(zip(result['trace'],record['step_inputs'])):
            if not entry.get('decision'):break
            if entry['response']['choices'][0]['message'].get('reasoning_content'):break
            expected=gold[j] if j<len(gold) else None
            actual={k:entry['proposal'][k] for k in ('name','arguments')} if entry['decision']=='call' else None
            current_hash=state_hash(sim.states);visible=json.dumps(inp['prompt'],ensure_ascii=False)
            candidates=[]
            if expected is None:
                if actual is not None and (Simulator.kind(actual['name'])=='write' or actual['name']=='sys.ask_missing_info'):
                    candidates.append(('stop',FINISH,callmsg(actual),'model_wrong_extra_action'))
                if gold:
                    previous=next((c for c in reversed(gold) if Simulator.kind(c['name'])=='write' and c['name']!='sys.ask_missing_info'),None)
                    if previous:candidates.append(('stop',FINISH,callmsg(previous),'verified_completed_goal_duplicate_write'))
            elif expected['name']=='sys.ask_missing_info':
                if actual is None or Simulator.kind(actual['name'])=='write':
                    candidates.append(('ask',callmsg(expected),FINISH if actual is None else callmsg(actual),'model_missing_information_error'))
                candidates.append(('ask',callmsg(expected),FINISH,'gold_requires_clarification'))
            else:
                if current_hash!=record['gold_final_state_hash']:
                    candidates.append(('continue',callmsg(expected),FINISH,'model_premature_finish' if actual is None else 'verified_unfinished_goal'))
                if actual is not None and actual['name']=='sys.ask_missing_info' and actual['arguments'].get('tool_name')==expected['name']:
                    params=actual['arguments'].get('params',[])
                    if params and all(k in expected['arguments'] and json.dumps(expected['arguments'][k],ensure_ascii=False).strip('"') in visible for k in params):
                        candidates.append(('ask',callmsg(expected),callmsg(actual),'model_unnecessary_ask_visible_evidence'))
                if actual is not None and actual['name']==expected['name'] and Simulator.kind(actual['name'])=='write':
                    a,b=expected['arguments'],actual['arguments']
                    changed={k for k in a if a.get(k)!=b.get(k)}
                    fields={'source','from_account_id','amount','time','schedule_time','start','enabled','power','steps','minutes','repeat'}
                    if set(a)==set(b) and changed and changed<=fields:
                        candidates.append(('argument',callmsg(expected),callmsg(actual),'model_same_tool_and_entity_wrong_decision_field'))
                if Simulator.kind(expected['name'])=='write':
                    # Clearly labelled synthetic negatives, independently reviewed
                    # before training. Keep tool/entity, perturb one scalar field.
                    # Do not count these as errors actually made by the model.
                    alternative=copy.deepcopy(expected);mutated=False
                    for field in ['amount','steps','minutes','time','schedule_time','start']:
                        value=alternative['arguments'].get(field)
                        if isinstance(value,(int,float)) and not isinstance(value,bool) and value>0:
                            alternative['arguments'][field]=value+max(1,int(abs(value)*.1));mutated=True
                        elif isinstance(value,str) and 'T' in value:
                            try:alternative['arguments'][field]=(datetime.fromisoformat(value)+timedelta(hours=1)).isoformat();mutated=True
                            except ValueError:pass
                        if mutated:break
                    if mutated and sim.validate(alternative['name'],alternative['arguments'])['ok']:
                        candidates.append(('argument',callmsg(expected),callmsg(alternative),'synthetic_single_scalar_argument_contrast'))
            for kind,chosen,rejected,origin in candidates:
                if chosen==rejected:continue
                pool.append({'id':f'{sid}:step{j}:{kind}:{len(pool)}','source_id':sid,'group_id':spec['group_id'],'stratum':spec['stratum'],
                    'kind':kind,'origin':origin,'model_error':origin.startswith('model_'),'prompt':inp['prompt'],'tools':inp['tools'],
                    'chosen':[copy.deepcopy(chosen)],'rejected':[copy.deepcopy(rejected)],'verified_gold_prefix_steps':j,
                    'prefix_state_hash':current_hash,'target_state_hash':record['gold_final_state_hash']})
            if expected is None or actual!=expected:break
            obs=sim.call(expected['name'],expected['arguments'])
            assert obs==entry['observation'],sid
    # De-duplicate candidates sharing the exact decision and pair.
    unique={}
    for e in pool:
        key=(e['source_id'],e['verified_gold_prefix_steps'],json.dumps(e['chosen'],sort_keys=True),json.dumps(e['rejected'],sort_keys=True))
        if key not in unique or e['model_error']:unique[key]=e
    pool=list(unique.values());rng=random.Random(20261115)
    for e in pool:e['_tie']=rng.random()
    selected=[];used=set();per_source=collections.Counter();counts=collections.Counter()
    # Equal top-level stop/continue/ask/argument signal, diverse source cap.
    for turn in range(16):
        for kind in ['continue','stop','ask','argument']:
            available=[e for e in pool if e['kind']==kind and e['id'] not in used and per_source[e['source_id']]<2]
            if not available:continue
            e=min(available,key=lambda e:(per_source[e['source_id']],not e['model_error'],e['_tie']))
            selected.append(e);used.add(e['id']);per_source[e['source_id']]+=1;counts[kind]+=1
    # Do not train an unbalanced stop-only or single-family pilot silently.
    assert len(selected)>=48 and counts['continue']>=12 and counts['stop']>=12 and counts['ask']>=8 and counts['argument']>=4,dict(counts)
    assert len(per_source)>=32
    for e in pool:e.pop('_tie',None)
    for e in selected:assert e['source_id'] in split['train_fit']
    rng.shuffle(selected)
    replay_path=ROOT/'llm/handoffs/claude_trainfit_review_20261114/sft/train_next_action.jsonl'
    replay=[e for e in rows(replay_path) if e['source_id'] not in set(EXCLUDED_SOURCES)];strata=collections.defaultdict(list)
    for e in replay:
        c=e['completion'][0].get('tool_calls',[]);strata[c[0]['function']['name'] if c else '__finish__'].append(e)
    n_old=3*len(selected);assert n_old>=len(strata)
    quota={k:max(1,int(n_old*len(v)/len(replay))) for k,v in strata.items()}
    while sum(quota.values())<n_old:
        k=max(strata,key=lambda k:n_old*len(strata[k])/len(replay)-quota[k]);quota[k]+=1
    while sum(quota.values())>n_old:
        k=max((k for k in quota if quota[k]>1),key=lambda k:quota[k]-n_old*len(strata[k])/len(replay));quota[k]-=1
    old=[]
    for k in sorted(strata):rng.shuffle(strata[k]);old+=strata[k][:quota[k]]
    rng.shuffle(old);assert len(old)==n_old
    write_rows(HERE/'data/preference_pool.jsonl',pool);write_rows(HERE/'data/preferences.jsonl',selected);write_rows(HERE/'data/replay.jsonl',old)
    shutil.copy2(replay_path,HERE/'data/full_replay_pool.jsonl');assert sha(replay_path)==sha(HERE/'data/full_replay_pool.jsonl')
    from transformers import AutoTokenizer
    from sft_data import encode
    tok=AutoTokenizer.from_pretrained(read(PARENT/'.runtime/MODELS.json')['snapshot'],local_files_only=True)
    token_pairs=[]
    for e in selected:
        ce=encode({**e,'completion':e['chosen']},tok);re=encode({**e,'completion':e['rejected']},tok)
        token_pairs.append({k:e[k] for k in ['id','source_id','group_id','kind','origin']}|{'chosen':ce,'rejected':re})
    write_rows(HERE/'data/preferences_tokens.jsonl',token_pairs)
    write_rows(HERE/'data/replay_tokens.jsonl',[encode(e,tok) for e in old])
    recipe={'seed':20261115,'steps':len(selected),'learning_rate':2e-6,'beta':.1,'replay_weight':1.,'replay_per_step':3,
        'dropout_disabled':True,'reference':'baseline50.9 original adapter, precomputed NF4/BF16 log probabilities',
        'rank':16,'alpha':32,'save_every_steps':10,'retain_all_checkpoints':True,
        'objective':'Step-DPO + mean of3 original SFT replay CE losses per optimizer update; exact memory-bounded DPO chain-rule gradient.',
        'no_VAL_or_public_training':True,'no_ASR_or_actor_changes':True,'no_external_upload':True,
        'public_gate':'Generate a local submission after valid complete trial/audits, even if DEV delta is negative; user chooses whether to upload.'}
    new(HERE/'recipe.json',recipe)
    new(HERE/'DATA.json',{'selected_pairs':len(selected),'pair_sources':len(per_source),'pair_kinds':dict(counts),
        'actual_model_error_pairs':sum(e['model_error'] for e in selected),'replay_rows':n_old,'replay_sources':len({e['source_id'] for e in old}),
        'replay_pool_rows':3906,'source_exclusions':dict(exclusions),'not_a_faithful_full_ETO_or_DAgger_reproduction':True,
        'replay_excluded_unresolved_TRAIN_sources':list(EXCLUDED_SOURCES),
        'pair_method':'Actual visited gold-matching prefixes; continuation/finish contrasts, selected first-error negatives, and explicitly labelled single-scalar synthetic negatives. Independent semantic review required; not every schema-valid alternative is semantically wrong.',
        'max_tokens':max(max(e['chosen']['length'],e['rejected']['length']) for e in token_pairs),
        'files_sha256':{str(p.relative_to(HERE)):sha(p) for p in (HERE/'data').glob('*.jsonl')}})
    print('PREFERENCES_VERIFIED',read(HERE/'DATA.json'),flush=True)
