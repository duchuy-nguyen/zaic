"""Check the exact public allowlist and obvious accidental private payloads."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run():
    manifest=json.loads((ROOT/'CODE_MANIFEST.json').read_text(encoding='utf8'))
    files={p.relative_to(ROOT).as_posix():p for p in ROOT.rglob('*') if p.is_file()}
    assert set(files)==set(manifest['files'])|{'CODE_MANIFEST.json','VALIDATION.json'},'Unexpected file in public folder'
    flags=[]
    for n,p in files.items():
        assert not p.is_symlink() and p.stat().st_size<2_000_000,n
        assert p.suffix.lower() not in {'.zip','.jsonl','.wav','.pt','.bin','.safetensors','.gguf','.dll','.exe'},n
        data=p.read_bytes();text=data.decode('utf8');assert '\x00' not in text,n
        if n in manifest['files']:assert hashlib.sha256(data).hexdigest()==manifest['files'][n],n
        if n!='audit_public.py':
            for pattern in (r'hf_[A-Za-z0-9]{20,}',r'gh[pousr]_[A-Za-z0-9]{20,}',r'sk-[A-Za-z0-9]{20,}',r'-----BEGIN .*PRIVATE KEY',r'train_\d{4}',r'[CD]:[\\/](?:Users|Desktop|UserProfileData)[\\/]'):
                if re.search(pattern,text):flags.append((n,pattern))
    assert not flags,flags
    print('PUBLIC_AUDIT_PASSED',len(files),'files')
if __name__=='__main__':run()
