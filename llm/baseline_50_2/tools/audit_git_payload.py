"""Check the shareable code tree contains no model, raw data or credentials."""
import argparse
from pathlib import Path
import re

DENY = {'.gguf', '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.h5', '.npz', '.npy',
        '.wav', '.mp3', '.flac', '.zip', '.7z', '.dll', '.exe', '.jsonl', '.pyc', '.log'}
PRIVATE_DIRS = {'private_assets', 'models', 'weights', 'checkpoints', 'outputs', 'logs', 'runtime', '.venv', '__pycache__'}
PATTERNS = [re.compile(r'\bhf_[A-Za-z0-9]{25,}\b'), re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}\b'),
            re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]


def audit(root):
    root = Path(root).resolve(); files = []
    for p in sorted(root.rglob('*')):
        if not p.is_file(): continue
        rel = p.relative_to(root)
        if p.suffix.lower() in DENY or set(rel.parts) & PRIVATE_DIRS or p.name.startswith('.env') or p.name == 'config.local.json':
            raise ValueError('Private/binary artifact inside Git code tree: ' + str(rel))
        if p.stat().st_size > 2 * 1024**2:
            raise ValueError('Unexpected large file in code tree: ' + str(rel))
        text = p.read_text(encoding='utf8')
        if any(pattern.search(text) for pattern in PATTERNS):
            raise ValueError('Possible credential in: ' + str(rel))
        if '\x00' in text:
            raise ValueError('Binary content in: ' + str(rel))
        files.append(rel.as_posix())
    return files


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args(); files = audit(args.root)
    print('CODE_ONLY_AUDIT_PASSED', len(files), 'text files; no model/data/binary/credential pattern found.')
