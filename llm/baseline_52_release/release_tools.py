"""Public payload audit and pinned private archive restoration. No uploads."""
import argparse,hashlib,json,re,shutil,uuid,zipfile
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def audit():
    m=read(HERE/'CODE_MANIFEST.json')['files'];found={p.relative_to(HERE).as_posix():p for p in HERE.rglob('*') if p.is_file()}
    assert set(found)==set(m)|{'CODE_MANIFEST.json','VALIDATION.json'},'Unexpected file in release'
    flags=[]
    for n,p in found.items():
        assert not p.is_symlink() and not p.is_junction() and p.stat().st_size<2_000_000,n
        assert p.suffix.lower() not in {'.zip','.jsonl','.pt','.pth','.bin','.safetensors','.gguf','.dll','.exe','.wav'},n
        payload=p.read_bytes();text=payload.decode('utf8');assert '\x00' not in text,n
        if n in m:assert hashlib.sha256(payload).hexdigest()==m[n],n
        if n.endswith('audit_public.py') or n=='release_tools.py':continue
        for pattern in (r'hf_[A-Za-z0-9]{20,}',r'gh[pousr]_[A-Za-z0-9]{20,}',r'sk-[A-Za-z0-9]{20,}',r'-----BEGIN .*PRIVATE KEY',r'train_\d{4}',r'[CD]:[\\/](?:Users|Desktop|UserProfileData)[\\/]'):
            if re.search(pattern,text):flags.append((n,pattern))
    assert not flags,flags
    print('PUBLIC_RELEASE_AUDIT_PASSED',len(found),'files')
def unpack(archive,dest):
    archive=Path(archive).resolve();dest=Path(dest).resolve();spec=read(HERE/'DRIVE_ARCHIVES.json')['files'][archive.name]
    assert archive.stat().st_size==spec['bytes'] and sha(archive)==spec['sha256'],'Wrong ZIP/hash'
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))
        metadata='PRIVATE_MANIFEST.json' if 'PRIVATE_MANIFEST.json' in names else 'ARCHIVE_MANIFEST.json'
        manifest=json.loads(z.read(metadata));files=manifest['files']
        assert set(names)==set(files)|{metadata},'Unexpected archive member'
        for n,info in files.items():
            rel=PurePosixPath(n);member=z.getinfo(n);target=dest.joinpath(*rel.parts).resolve()
            assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in n and ':' not in n and target.is_relative_to(dest),n
            assert ((member.external_attr>>16)&0o170000)!=0o120000,n
            if target.exists():assert target.is_file() and target.stat().st_size==info['bytes'] and sha(target)==info['sha256'],n;continue
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.part')
            with z.open(n) as source,tmp.open('xb') as out:shutil.copyfileobj(source,out,8*1024**2)
            assert tmp.stat().st_size==info['bytes'] and sha(tmp)==info['sha256'],n
            tmp.rename(target)
        # Keep parent PRIVATE_MANIFEST in its original shape, used by the training verifier.
        target=dest/metadata
        if metadata=='PRIVATE_MANIFEST.json':
            payload=z.read(metadata)
            if target.exists():assert target.read_bytes()==payload
            else:target.write_bytes(payload)
        log=dest/'.archive_manifests';log.mkdir(exist_ok=True);target=log/(archive.name+'.json');payload=z.read(metadata)
        if target.exists():assert target.read_bytes()==payload
        else:target.write_bytes(payload)
    print('UNPACKED_AND_VERIFIED',archive.name)
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True);sub.add_parser('audit')
    u=sub.add_parser('unpack');u.add_argument('--archive',required=True);u.add_argument('--dest',required=True)
    a=p.parse_args();audit() if a.stage=='audit' else unpack(a.archive,a.dest)
if __name__=='__main__':main()
