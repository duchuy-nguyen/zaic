"""Strict public code payload audit. It reads files; it never adds or pushes Git."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

DENY_SUFFIXES = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx",
    ".h5", ".npz", ".npy", ".wav", ".mp3", ".flac", ".m4a", ".mp4", ".zip", ".rar",
    ".tar", ".gz", ".7z", ".dll", ".exe", ".jsonl", ".pyc", ".pyo", ".log", ".key", ".pem"}
PRIVATE_DIRS = {"private_assets", "official", "data", "datasets", "models", "weights",
    "checkpoints", "outputs", "records", "logs", "runtime", ".runtime", ".venv", "venv", "__pycache__", ".git"}
PATTERNS = [
    re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{25,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}\b"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"[A-Za-z]:[/\\](?:Users|Desktop|Downloads|UserProfileData)[/\\]"),
    re.compile(r"\b(?:train|public_test)_\d{4}\b"),
    re.compile(r"(?m)^version https://git-lfs.github.com/spec/v1\s*$"),
]

def sha(path):
    with path.open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()

def audit(root):
    root=Path(root).resolve(); manifest=json.loads((root/"CODE_MANIFEST.json").read_text(encoding="utf8"))
    expected=set(manifest["files_sha256"])|{"CODE_MANIFEST.json","PACKAGING_VERIFICATION.json"}
    found=[]; total=0
    for path in sorted(root.rglob("*")):
        rel=path.relative_to(root)
        if path.is_symlink() or getattr(path,"is_junction",lambda:False)():
            raise ValueError("Link/junction is forbidden: "+str(rel))
        if set(rel.parts)&PRIVATE_DIRS:
            raise ValueError("Private/generated directory in public tree: "+str(rel))
        if not path.is_file(): continue
        if path.suffix.lower() in DENY_SUFFIXES or path.name.startswith(".env") or path.name=="config.local.json":
            raise ValueError("Private/binary artifact: "+str(rel))
        if path.stat().st_size>2*1024**2: raise ValueError("Unexpected large file: "+str(rel))
        content=path.read_text(encoding="utf8")
        if "\x00" in content or any(pattern.search(content) for pattern in PATTERNS):
            raise ValueError("Possible binary/credential/private content: "+str(rel))
        name=rel.as_posix()
        if name not in expected: raise ValueError("Unreviewed file: "+name)
        if name in manifest["files_sha256"] and sha(path)!=manifest["files_sha256"][name]:
            raise ValueError("Changed reviewed file: "+name)
        total+=path.stat().st_size; found.append(name)
    missing=expected-set(found)
    if missing: raise ValueError("Missing reviewed files: "+str(sorted(missing)))
    return {"passed":True,"files":len(found),"bytes":total,
            "weights_data_binary_credentials_detected":False,"manifest_matches":True}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();print(json.dumps(audit(args.root),indent=2))
