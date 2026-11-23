"""Explicit DPO reproduction; no training is started by packaging/import/help."""
import argparse,hashlib,importlib,importlib.util,importlib.metadata,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PUBLIC=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst),dst
    else:shutil.copyfile(src,dst)
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
    if p.exists():assert p.read_text(encoding='utf8')==payload,p
    else:p.write_text(payload,encoding='utf8')
def config(path):
    c=read(path)
    for k in ('private_dpo_dir','private_base_dir','inference_assets_dir','work_dir','qwen_base_dir','asr_base_dir'):
        c[k]=str(Path(c[k]).expanduser().resolve())
    assert c['work_dir'] not in (c['private_dpo_dir'],c['private_base_dir'],c['inference_assets_dir'])
    return c
def snapshot(c):
    pins=read(PUBLIC/'training/base50_9/BASE_MODELS.json')['qwen'];base=Path(c['qwen_base_dir'])
    for n,v in pins['files'].items():assert sha(base/n)==v['sha256'],n
def dataset_check(c):
    p=Path(c['private_dpo_dir']);contract=read(HERE/'DATA_CONTRACT.json')
    assert sha(p/'PAIR_REVIEW.json')==contract['review_sha256']
    for n,h in contract['data_files_sha256'].items():assert sha(p/n)==h,n
    review=read(p/'PAIR_REVIEW.json');ids=set(review['approved_ids'])
    pairs=[json.loads(l) for l in (p/'data/preferences_tokens.jsonl').read_text(encoding='utf8').splitlines()]
    replay=[json.loads(l) for l in (p/'data/replay_tokens.jsonl').read_text(encoding='utf8').splitlines()]
    selected=[j for j,r in enumerate(pairs) if r['id'] in ids]
    assert len(ids)==len(selected)==63 and len(replay)==192
    split=read(Path(c['private_base_dir'])/'prepared/split.json');fit=set(split['train_fit']);val=set(split['val_main'])
    active=[pairs[j] for j in selected]+[r for j in selected for r in replay[j*3:j*3+3]]
    assert len(active)==252 and all(r['source_id'] in fit and r['source_id'] not in val for r in active)
    assert not {split['group_ids'][r['source_id']] for r in active}&{split['group_ids'][i] for i in val}
    assert len({r['source_id'] for r in active})==236
    return {'passed':True,'draft_pairs':len(pairs),'approved_pairs':63,'approved_replay_rows':189,'unique_active_sources':236,'no_VAL_gradient_sources':True}
def setup(c,rebuild=False):
    dataset_check(c);snapshot(c);work=Path(c['work_dir']);assert not work.exists(),'Choose a NEW work folder'
    p=Path(c['private_dpo_dir']);b=Path(c['private_base_dir']);assets=Path(c['inference_assets_dir']);work.mkdir(parents=True)
    for folder in ('inputs','harvest'):
        source=p/folder
        for f in source.rglob('*'):
            if f.is_file():put(f,work/folder/f.relative_to(source))
    for n in ('PAIR_REVIEW.json','PRESERVATION.json','RUNTIME.json'):
        put(p/n,work/n)
    if not rebuild:
        shutil.copytree(p/'data',work/'data');put(p/'recipe.json',work/'recipe.json');put(p/'DATA.json',work/'DATA.json')
        metadata=read(work/'DATA.json')
        metadata['files_sha256']={name.replace('\\','/'):digest for name,digest in metadata['files_sha256'].items()}
        # Workspace metadata only: keep the immutable private ZIP and all training rows unchanged.
        (work/'DATA.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    parent=work/'parent';save(parent/'.runtime/MODELS.json',{'snapshot':c['qwen_base_dir']})
    # Exact SFT replay pool reused by original prepare.build, all contents private.
    put(b/'prepared/train.jsonl',work/'sandbox/llm/handoffs/claude_trainfit_review_20261114/sft/train_next_action.jsonl')
    control=work/'control/assets_large'
    for n in ('retrieval_index.json','split.json'):
        put(assets/'private_data'/n,control/'private_data'/n)
    for f in (assets/'runtime').iterdir():
        if f.is_file():put(f,control/'runtime'/f.name)
    source=read(p/'SOURCE.json');source.update(baseline_model=str(next((p/'parent_qwen').glob('*.gguf'))),
        catalog=str(p/'train_inputs/train_tools_catalog.json'),split=str(p/'train_inputs/split_train_only.json'),training_sdk=str(p/'sdk/train'))
    paths=[p/'train_inputs/train_fit.jsonl',p/'train_inputs/split_train_only.json',p/'train_inputs/train_tools_catalog.json',work/'inputs/plan.json',work/'inputs/train_sources.jsonl']
    source['external_sha256']={str(f):sha(f) for f in paths};save(work/'SOURCE.json',source)
    (work/'data').mkdir(exist_ok=True);(work/'outputs').mkdir(exist_ok=True)
    save(work/'PORTABLE_CONFIG.json',c)
    if not rebuild:freeze(work)
    print('SETUP_READY',work)
def activate(c):
    work=Path(c['work_dir']);os.environ['LIFEMATE_DPO_REPRO_CONFIG']=str(work/'PORTABLE_CONFIG.json')
    sys.path.insert(0,str(HERE/'dpo_core'));sys.path.insert(1,str(PUBLIC/'inference'))
def freeze(work):
    names=('DATA.json','PAIR_REVIEW.json','PRESERVATION.json','recipe.json','SOURCE.json','PORTABLE_CONFIG.json','parent/.runtime/MODELS.json')
    save(work/'FREEZE.json',{'files_sha256':{n:sha(work/n) for n in names},'portable_source_guard':'release root CODE_MANIFEST; no unrelated old experiment files'})
def rebuild(c):
    setup(c,rebuild=True);activate(c)
    import prepare
    prepare.build()
    p=Path(c['private_dpo_dir']);work=Path(c['work_dir']);contract=read(HERE/'DATA_CONTRACT.json')
    for n,h in contract['data_files_sha256'].items():assert sha(work/n)==h,n
    freeze(work);save(work/'REBUILT_DATA_PARITY.json',{'passed':True,'draft_pairs':64,'review_approved_pairs':63,'all_six_training_data_files_byte_equal':True})
    print('DPO_DATA_RECONSTRUCTION_BYTE_PARITY_PASSED')
def reference_cache(c):
    versions()
    dataset_check(c);work=Path(c['work_dir']);p=Path(c['private_dpo_dir'])
    assert not (work/'outputs/reference').exists(),'Preserve reference outputs'
    shutil.copytree(p/'outputs/reference',work/'outputs/reference')
    print('HISTORICAL_REFERENCE_CACHE_RESTORED: original NF4/BF16 stack only; recompute for changed stack')
def versions():
    expected=read(PUBLIC/'training/base50_9/ENVIRONMENT.json')['required_training_packages']
    for n,v in expected.items():
        assert importlib.metadata.version(n)==v,'Historical training stack requires '+n+'=='+v
def merge_export(c):
    # Reuse the base package's explicit merge/export implementation, select the final52 adapter.
    p=Path(c['private_dpo_dir']);base=Path(c['private_base_dir']);work=Path(c['work_dir'])/'final52_export'
    assert not work.exists(),'Choose a new output work root';work.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('parent_repro52_export',PUBLIC/'training/base50_9/reproduce.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg={'private_dir':base,'work_dir':work,'qwen_base_dir':c['qwen_base_dir'],'asr_base_dir':c['asr_base_dir'],
        'qwen_processor_dir':str(p/'outputs/decision_dpo/tokenizer')}
    for key in ('quantizer','quantizer_sha256'):
        if key in c:cfg[key]=c[key]
    module.merge(cfg,'qwen',str(p/'outputs/decision_dpo/adapter_final'));module.export(cfg)
    print('EXPORTED52_FROM_SHIPPED_FINAL_ADAPTER',work)
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify-data','setup','rebuild-data','restore-reference','math','reference','smoke','train','harvest','merge-export'])
    p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();c=config(a.config)
    if a.stage=='verify-data':print(json.dumps(dataset_check(c)));return
    if a.stage=='setup':setup(c);return
    if a.stage=='rebuild-data':rebuild(c);return
    if a.stage=='restore-reference':reference_cache(c);return
    if a.stage=='merge-export':merge_export(c);return
    activate(c)
    if a.stage=='harvest':
        import prepare
        prepare.harvest(a.resume);return
    # Historical recovery changed allocator only for training smoke/full, not inference or reference.
    if a.stage in ('smoke','train'):os.environ['PYTORCH_CUDA_ALLOC_CONF']='backend:cudaMallocAsync'
    if a.stage in ('reference','smoke','train'):versions()
    import train_dpo as trainer
    if a.stage=='math':trainer.math_test()
    elif a.stage=='reference':trainer.reference(a.resume)
    else:trainer.train(a.stage=='smoke',a.resume)
if __name__=='__main__':main()
