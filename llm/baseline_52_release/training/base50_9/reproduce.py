"""Explicit offline data/train/merge/export commands. No action occurs on import."""
import argparse, hashlib, importlib, importlib.metadata, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if p.exists():
        if p.read_text(encoding='utf8')!=payload:raise ValueError('Preserve existing artifact: '+str(p))
    else:p.write_text(payload,encoding='utf8')
def source_check():
    m=read(HERE/'CODE_MANIFEST.json')
    for n,h in m['files'].items():
        if sha(HERE/n)!=h:raise ValueError('Public source changed: '+n)
def config(p):
    c=read(p)
    for n in ('private_dir','work_dir'):c[n]=Path(c[n]).expanduser().resolve()
    if c['private_dir']==c['work_dir']:raise ValueError('Use separate private input and new work directory')
    c['work_dir'].mkdir(parents=True,exist_ok=True)
    return c
def private_check(c,full=False):
    source_check();root=c['private_dir'];expected=read(HERE/'EXPECTED_ARTIFACTS.json')
    if sha(root/'PRIVATE_MANIFEST.json')!=expected['private_manifest_sha256']:raise ValueError('Wrong private bundle')
    m=read(root/'PRIVATE_MANIFEST.json')
    wanted=m['files'] if full else expected['files']
    for n,v in wanted.items():
        p=root/n
        if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('Private input changed: '+n)
    split=read(root/'prepared/split.json')
    a,b=set(split['train_fit']),set(split['val_main'])
    assert len(a)==1236 and len(b)==309 and not a&b
    assert not {split['group_ids'][i] for i in a}&{split['group_ids'][i] for i in b}
    for part,ids,n in [('train',a,3906),('val',b,963)]:
        rows=[json.loads(l) for l in (root/'prepared'/f'{part}.jsonl').read_text(encoding='utf8').splitlines()]
        assert len(rows)==n and {r['source_id'] for r in rows}==ids
    return m
def unpack(archive,dest,checksums):
    archive=Path(archive).resolve();dest=Path(dest).resolve()
    expected=read(checksums)['files'][archive.name]
    assert archive.stat().st_size==expected['bytes'] and sha(archive)==expected['sha256']
    assert not dest.exists(),'Choose a NEW private destination'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))
        for n in names:
            p=Path(n);target=(dest/p).resolve()
            assert not p.is_absolute() and ':' not in n and '\\' not in n and target.is_relative_to(dest),n
        for i in z.infolist():
            if (i.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlink archive member')
        z.extractall(dest)
    print('UNPACKED',dest)
def prepare(c):
    private_check(c);root=c['private_dir'];out=c['work_dir']/'prepared_rebuilt'
    assert not out.exists(),'Keep previous generation; choose a new work directory'
    from train_core import prepare as p
    p.HERE=root/'prepared';p.SOURCE_SPLIT=root/'prepared/split.json';p.RAW_TRAIN=root/'raw'
    p.NOISE_SOURCE=root/'augmentation/train_only_noise.jsonl'
    p.build(out)
    # Metadata changes record portable paths/source/filtered noise. Actual model inputs must match byte for byte.
    names=('train.jsonl','val.jsonl','train_asr_clips.json','val_asr_clips.json','retrieval_index.json','split.json','SYSTEM.txt','config.json')
    parity={n:sha(out/n)==sha(root/'prepared'/n) for n in names}
    assert all(parity.values()),parity
    save(c['work_dir']/'PREPARE_PARITY.json',{'passed':True,'byte_equal_model_inputs':parity,
        'metadata_note':'DATA_MANIFEST records changed portable source hashes and TRAIN-filtered noise provenance; not byte equal to original.'})
def snapshot(c,role):
    folder=Path(c[role+'_base_dir']).resolve();pin=read(HERE/'BASE_MODELS.json')[role]
    for n,v in pin['files'].items():
        if sha(folder/n)!=v['sha256']:raise ValueError('Base snapshot mismatch: '+role+'/'+n)
    return folder
def train_module(c,role):
    private_check(c);base=snapshot(c,role);root=c['private_dir'];work=c['work_dir']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128,garbage_collection_threshold:0.8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false');os.environ.setdefault('WANDB_DISABLED','true')
    recipe=read(HERE/'training_recipe.json')
    # Version mismatch is visible before loading GPU weights. Never silently substitute a new training API.
    for n,v in read(HERE/'ENVIRONMENT.json')['required_training_packages'].items():
        if importlib.metadata.version(n)!=v:raise RuntimeError('Training requires '+n+'=='+v)
    if role=='qwen':
        m=importlib.import_module('train_core.qwen35_training')
        m.HERE=work/'qwen';m.HERE.mkdir(exist_ok=True);m.LIFE=root;m.ROOT=work
        m.model_snapshot=lambda:base
        def guard(data,full=False):
            private_check(c)
            manifest=read(root/'prepared/DATA_MANIFEST.json')
            assert manifest['config']['llm']['base']==recipe['qwen']['base']
            assert manifest['config']['llm']['revision']==recipe['qwen']['revision']
            return manifest
        m.verify=guard
    else:
        m=importlib.import_module('train_core.phowhisper_large_training')
        m.HERE=work/'asr';m.HERE.mkdir(exist_ok=True);m.ROOT=work
        # Rebind ONLY the historical dataset root; feature extraction and collator functions remain exact.
        original=m.asr_dataset
        import ast
        src=Path(m.__file__).read_text(encoding='utf8');node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='asr_dataset')
        fn=ast.get_source_segment(src,node)
        old="base = ROOT / 'LifeMate/data/lifemate-v3.2.0/train+public_test/train'"
        assert old in fn;fn=fn.replace(old,'base = RAW_TRAIN')
        env=dict(original.__globals__);env['RAW_TRAIN']=root/'raw';exec(compile(fn,'<portable-asr-dataset-root>','exec'),env)
        m.asr_dataset=env['asr_dataset'];m.snapshot=lambda kind:base
        def guard():
            private_check(c)
            return {'asr':recipe['asr'],'asr_actual_trainer':recipe['asr']['actual_trainer']}
        m.verify_data=guard;m.verify_frozen=source_check
        # Training worker expects these input names directly below HERE/data.
        data=m.HERE/'data';data.mkdir(exist_ok=True)
        for n in ('train_asr_clips.json','val_asr_clips.json'):
            target=data/n
            if target.exists():assert sha(target)==sha(root/'prepared'/n)
            else:shutil.copyfile(root/'prepared'/n,target)
        save(m.HERE/'DATA_PARITY.json',{'portable_guard':'PRIVATE_MANIFEST + CODE_MANIFEST; no external unrelated baseline files'})
        save(m.HERE/'recipe.json',guard())
    return m
def profile(c):
    m=train_module(c,'qwen');out=c['work_dir']/'qwen/outputs';out.mkdir(exist_ok=True)
    m.profile(c['private_dir']/'prepared',out)
    p=read(out/'TOKEN_PROFILE.json');old=read(c['private_dir']/'tokens/TOKEN_PROFILE.json')
    for part in ('train','val'):assert p['partitions'][part]['token_cache_sha256']==old['partitions'][part]['token_cache_sha256']
    save(c['work_dir']/'TOKEN_PARITY.json',{'passed':True})
def train(c,role,smoke,resume):
    m=train_module(c,role)
    if role=='qwen':m.training(c['private_dir']/'prepared',c['work_dir']/'qwen/outputs',smoke,resume)
    else:m.training(smoke,resume,'portable')
def merge(c,role,adapter):
    private_check(c);base=snapshot(c,role)
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,WhisperForConditionalGeneration,AutoTokenizer,WhisperProcessor
    from peft import PeftModel
    if adapter:
        adapter=Path(adapter).resolve()
        processor=Path(c[role+'_processor_dir']).resolve()
    else:adapter=c['private_dir']/'adapters'/role;processor=c['private_dir']/'processors'/role
    out=c['work_dir']/'merged'/role
    assert not out.exists(),'Preserve previous merge outputs'
    cls=Qwen3_5ForConditionalGeneration if role=='qwen' else WhisperForConditionalGeneration
    model=cls.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16 if role=='qwen' else torch.float16,device_map={'':'cpu'})
    model=PeftModel.from_pretrained(model,adapter,device_map={'':'cpu'}).merge_and_unload(safe_merge=True)
    model.config.use_cache=True
    model.save_pretrained(out,safe_serialization=True,max_shard_size='2GB')
    (AutoTokenizer if role=='qwen' else WhisperProcessor).from_pretrained(processor,local_files_only=True).save_pretrained(out)
    save(out/'MERGE_PROVENANCE.json',{'role':role,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),
        'base_revision':read(HERE/'BASE_MODELS.json')[role]['revision'],'parameters':sum(p.numel() for p in model.parameters())})
def export(c):
    private_check(c);converter=c['private_dir']/'converter';pin=read(HERE/'CONVERTER_PIN.json')
    for n,h in pin['files'].items():assert sha(converter/n)==h,n
    source=c['work_dir']/'merged/qwen';assert (source/'MERGE_PROVENANCE.json').is_file()
    out=c['work_dir']/'exports';assert not out.exists(),'Preserve previous export outputs'
    out.mkdir();full=out/'Qwen3.5-4B-LifeMate-V2-BF16.gguf';quant=out/'Qwen3.5-4B-LifeMate-V2-Q4_K_M.gguf'
    subprocess.run([sys.executable,'-B',str(converter/'convert_hf_to_gguf.py'),str(source),'--outfile',str(full),'--outtype','bf16','--no-nextn'],cwd=converter,check=True)
    exe=Path(c.get('quantizer',str(c['private_dir']/'runtime_windows/llama-quantize.exe'))).resolve()
    pin=c.get('quantizer_sha256','20be1750e4b41ab927a48179c2ed7cea7b53d68307d1a8aed8117962a5fe5557')
    assert sha(exe)==pin,'Wrong quantizer: configure and record a platform-specific pinned binary explicitly'
    subprocess.run([str(exe),str(full),str(quant),'Q4_K_M'],check=True)
    save(out/'EXPORT_MANIFEST.json',{'bf16_sha256':sha(full),'q4_sha256':sha(quant),'quantizer_sha256':sha(exe),'converter_release':'b11800'})
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['unpack','verify','prepare','profile','train-qwen','train-asr','merge-qwen','merge-asr','export','retrieval'])
    p.add_argument('--config');p.add_argument('--archive');p.add_argument('--dest');p.add_argument('--checksums')
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--adapter-dir')
    args=p.parse_args()
    if args.stage=='unpack':unpack(args.archive,args.dest,args.checksums);return
    if not args.config:p.error('--config is required')
    c=config(args.config)
    if args.stage=='verify':private_check(c,full=True);print('VERIFIED: source, exact private payload, TRAIN/VAL groups')
    elif args.stage=='prepare':prepare(c)
    elif args.stage=='profile':profile(c)
    elif args.stage.startswith('train-'):train(c,args.stage.split('-')[1],args.smoke,args.resume)
    elif args.stage.startswith('merge-'):merge(c,args.stage.split('-')[1],args.adapter_dir)
    elif args.stage=='export':export(c)
    else:
        private_check(c);p=c['work_dir']/'prepared_rebuilt/retrieval_index.json'
        assert p.is_file(),'Run prepare first; retrieval is generated from TRAIN-fit by that stage'
        assert sha(p)==sha(c['private_dir']/'prepared/retrieval_index.json');print('RETRIEVAL_VERIFIED',p)
if __name__=='__main__':main()
