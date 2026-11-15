"""Historical hashing and checkpoint-artifact helpers."""
import hashlib,json,os
from pathlib import Path

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def new(p, text):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_bytes() == text.encode('utf8'), 'Preserve different existing artifact: ' + str(p)
    else:
        with p.open('xb') as f:
            f.write(text.encode('utf8'))

def json_new(p, obj):
    new(p, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def atomic(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.' + str(os.getpid()) + '.tmp')
    q.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    q.replace(p)
