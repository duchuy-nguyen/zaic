"""Freeze notebook-inspired TRAIN-fit data. VAL is clean, group-disjoint, never augmented."""
from collections import Counter
import copy,hashlib,json,shutil
from pathlib import Path
from policy import VERSION,SYSTEM,selected,namespace_mask,compact,normalize
from full_sft import make_example
HERE=Path(__file__).resolve().parent
LLM=HERE.parents[1];LIFE=LLM.parent/'LifeMate'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def new(p,s):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=s.encode('utf8')
    if p.exists():assert p.read_bytes()==b,'Refuse overwrite: '+str(p)
    else:p.write_bytes(b)
def json_new(p,x):new(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def build(output):
    cfg=read(HERE/'config.json');runtime=read(HERE/'RUNTIME_PATHS.json');old=Path(runtime['v2_root'])/'data/bundle_v1'
    split_path=LIFE/'data/manifests/split-v1.json';split=read(split_path)
    base=LIFE/'data/lifemate-v3.2.0/train+public_test/train'
    assert sha(base/'train.jsonl')==split['train_jsonl_sha256']
    rows={r['id']:r for r in map(json.loads,(base/'train.jsonl').read_text(encoding='utf8').splitlines())}
    assert len(split['train_fit'])==1236 and len(split['val_main'])==309
    assert not {split['group_ids'][i] for i in split['train_fit']}&{split['group_ids'][i] for i in split['val_main']}
    catalog={t['name']:t for a in read(base/'train_tools_catalog.json')['apps'] for t in a['tools']}
    noise_root=HERE/'data/selected_medium_noise';noise_state=read(noise_root/'MANIFEST.json')
    assert noise_state['status']=='completed'
    noises=[json.loads(l) for l in (noise_root/'transcripts.jsonl').read_text(encoding='utf8').splitlines()]
    noise={r['audio']:r for r in noises};noise_by_sha={r['audio_sha256']:r for r in noises};assert len(noise)==noise_state['total_clips']
    demos=[]
    for i in split['train_fit']:
        r=rows[i];e=make_example(r,catalog)
        demo={'context':{k:v for k,v in r['context'].items() if k!='installed_apps'},'request':r['audio_text'],
              'history':[{k:t[k] for k in ('role','audio_text','text','tool_calls','tool_results') if k in t} for t in r.get('history',[])],
              'trajectory':e['messages'][e['supervision_start']:]}
        if len(compact(demo))<=2200:
            demos.append(dict(source_id=i,group_id=split['group_ids'][i],request=r['audio_text'],demo=demo,
                              tool_names=sorted({c['name'] for t in r.get('history',[])+[r['label']] for c in t.get('tool_calls',[])})))
    assert demos
    stats=Counter();manifest=dict(version=cfg['version'],policy=VERSION,config=cfg,partitions={},
        train_sha256=sha(base/'train.jsonl'),split_sha256=sha(split_path),catalog_sha256=sha(base/'train_tools_catalog.json'),
        noise_source_sha256=sha(noise_root/'transcripts.jsonl'),noise_manifest_sha256=sha(noise_root/'MANIFEST.json'),
        retrieval_train_sources=len(demos),no_public_test=True,no_hidden_state_inputs=True)
    for part,key in [('train','train_fit'),('val','val_main')]:
        examples=[]
        for i in split[key]:
            row=copy.deepcopy(rows[i]);cat=catalog
            if part=='train':
                for j,t in enumerate([row]+[h for h in row.get('history',[]) if h['role']=='user']):
                    if selected(i,f'noise-turn-{j}',.3):
                        stats['noise_selected_turns']+=1;n=noise.get(t['audio']) or noise_by_sha[sha(base/t['audio'])]
                        if n['status']=='completed':
                            noisy=normalize(n['text']);stats['noise_changed_turns']+=noisy!=normalize(t['audio_text'])
                            t['audio_text']=noisy
                        else:stats['noise_excluded_failed_turns']+=1
                if selected(i,'namespace-mask',.5):
                    row,cat,mapping=namespace_mask(row,catalog)
                    stats['masked_conversations']+=1;stats['masked_namespaces']+=len(mapping)
            examples.append(make_example(row,cat))
        assert {e['source_id'] for e in examples}==set(split[key])
        target=output/(part+'.jsonl');new(target,''.join(compact(e)+'\n' for e in examples))
        manifest['partitions'][part]=dict(source_ids=split[key],sources=len(split[key]),examples=len(examples),sha256=sha(target))
        clip_file=old/(part+'_asr_clips.json');new(output/clip_file.name,clip_file.read_text(encoding='utf8'))
    assert not {r['sha256'] for r in read(output/'train_asr_clips.json')}&{r['sha256'] for r in read(output/'val_asr_clips.json')}
    assert stats['masked_conversations']>400 and stats['noise_changed_turns']>0
    json_new(output/'retrieval_index.json',demos);json_new(output/'split.json',split)
    new(output/'config.json',(HERE/'config.json').read_text(encoding='utf8'));new(output/'SYSTEM.txt',SYSTEM)
    manifest['augmentation_counts']=dict(stats)
    manifest['feature_source_sha256']={n:sha(HERE/n) for n in ('policy.py','full_sft.py','prepare.py','config.json','vnnum_notebook.py')}
    json_new(output/'DATA_MANIFEST.json',manifest)
    print('DATA_READY',dict(stats),'demos',len(demos),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);build(p.parse_args().output.resolve())
