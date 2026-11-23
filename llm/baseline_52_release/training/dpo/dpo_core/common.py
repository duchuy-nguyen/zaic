"""Portable private roots; retain model/data/source guards without old machine paths."""
import hashlib,json,os,sys,time,uuid
from pathlib import Path
PUBLIC=Path(__file__).resolve().parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
CONFIG=read(os.environ['LIFEMATE_DPO_REPRO_CONFIG'])
HERE=Path(CONFIG['work_dir']).resolve()
ROOT=HERE/'sandbox';PARENT=HERE/'parent';CONTROL=HERE/'control'
PRIVATE=Path(CONFIG['private_dpo_dir']).resolve();BASE=Path(CONFIG['private_base_dir']).resolve()
ADAPTER=BASE/'adapters/qwen'
EXCLUDED_SOURCES=read(PRIVATE/'train_inputs/excluded_sources.json')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(p)
def new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert read(p)==v,p;return
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def verify_preservation():
    assert sha(ADAPTER/'adapter_model.safetensors')=='7d04b7fd5feea6b92c230f4e710df25046a710c5d80ac9d8fe1f4e6dd73c93a5'
    p=read(HERE/'PRESERVATION.json')
    for n,h in p['actor_policy_sha256'].items():
        if n=='io.py':continue # platform/package wrapper, not an actor decision change
        assert sha(PUBLIC/'inference/baseline50'/n)==h,n
    return p
def verify():
    verify_preservation()
    for n,h in read(PUBLIC/'CODE_MANIFEST.json')['files'].items():assert sha(PUBLIC/n)==h,n
    for n,h in read(HERE/'FREEZE.json')['files_sha256'].items():assert sha(HERE/n)==h,n
    contract=read(PUBLIC/'training/dpo/DATA_CONTRACT.json')
    assert sha(HERE/'PAIR_REVIEW.json')==contract['review_sha256']
    for n,h in contract['data_files_sha256'].items():assert sha(HERE/n)==h,n
    pin=read(PUBLIC/'training/base50_9/BASE_MODELS.json')['qwen']
    snapshot=Path(read(PARENT/'.runtime/MODELS.json')['snapshot'])
    for n,v in pin['files'].items():assert sha(snapshot/n)==v['sha256'],n
    return read(HERE/'PRESERVATION.json')
