"""Configuration, atomic checkpoints and sanitized official inputs."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    until = time.monotonic() + 15
    while True:
        try:
            temp.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= until:
                raise
            time.sleep(.1)


def new_text(path, text):
    path = Path(path)
    payload = text.encode('utf8')
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Preserve different existing output: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(payload)


def acquire_lock(path):
    f = Path(path).open('a+b')
    f.seek(0)
    if not f.read(1):
        f.write(b'0'); f.flush()
    f.seek(0)
    try:
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        f.close()
        raise
    return f


def load_config(path):
    path = Path(path).resolve()
    data = read(path)
    if set(data) != {'assets_dir', 'dataset_dir', 'output_dir', 'llama_server'}:
        raise ValueError('Use the four path fields in config.example.json; baseline policy is frozen.')
    return {k: (path.parent / v).resolve() for k, v in data.items()}


def verify_assets(config):
    manifest_path = Path(__file__).parent / 'assets.manifest.json'
    manifest = read(manifest_path)
    for relative, info in manifest['files'].items():
        p = (config['assets_dir'] / relative).resolve()
        if not p.is_relative_to(config['assets_dir']):
            raise ValueError('Asset path escapes private_assets')
        if not p.is_file() or p.stat().st_size != info['bytes'] or sha(p) != info['sha256']:
            raise ValueError('Missing or changed Drive asset: ' + relative)
    server_info = manifest['files']['runtime/llama-server.exe']
    if sha(config['llama_server']) != server_info['sha256']:
        raise ValueError('This release pins the verified Windows CUDA b11800 server. Different platforms need a separately validated variant.')
    return manifest


def safe_row(row):
    return {'id': row['id'], 'audio': row['audio'], 'context': row['context'], 'history': [
        {'role': 'user', 'audio': t['audio']} if t['role'] == 'user' else
        {k: t[k] for k in ('role', 'text', 'tool_calls', 'tool_results') if k in t}
        for t in row.get('history', [])]}


def load_inputs(config, mode, manifest):
    split = 'train' if mode == 'val' else 'public_test'
    root = config['dataset_dir'] / split
    data = root / (split + '.jsonl')
    spec = manifest['dataset'][split]
    if sha(data) != spec['jsonl_sha256'] or sha(root / (split + '_tools_catalog.json')) != spec['catalog_sha256']:
        raise ValueError('Dataset/catalog differs from the frozen baseline release')
    runtime = {p.relative_to(root / 'zaic_tools').as_posix(): sha(p) for p in sorted((root / 'zaic_tools').rglob('*.py'))}
    if runtime != spec['runtime_py_sha256']:
        raise ValueError('Official ToolEnv SDK differs from the frozen release')
    rows = [json.loads(line) for line in data.read_text(encoding='utf8').splitlines() if line.strip()]
    if mode == 'public':
        for row in rows:
            if set(row) != {'id', 'audio', 'context', 'history'}:
                raise ValueError('Unexpected public fields; labels/transcripts are forbidden')
            if any(t['role'] == 'user' and 'audio_text' in t for t in row['history']):
                raise ValueError('Public gold transcript is forbidden')
        if len(rows) != 634 or len({r['id'] for r in rows}) != 634:
            raise ValueError('Expected 634 unique public IDs')
    else:
        split_manifest = read(config['assets_dir'] / 'private_data/split.json')
        by_id = {r['id']: r for r in rows}
        rows = [by_id[i] for i in split_manifest['val_main']]
        if len(rows) != 309 or len({r['id'] for r in rows}) != 309:
            raise ValueError('Expected the frozen 309 VAL IDs')
    catalog_data = read(root / (split + '_tools_catalog.json'))
    tools = {t['name']: t for a in catalog_data['apps'] for t in a.get('tools', [])}
    return [safe_row(r) for r in rows], tools


def resume_records(output, ids):
    directory = output / 'records'
    directory.mkdir(exist_ok=True)
    committed = {p.stem: read(p) for p in directory.glob('*.json')}
    if set(committed) != set(ids[:len(committed)]):
        raise ValueError('Non-prefix/unknown records; preserve artifacts and investigate')
    inflight = output / 'inflight.json'
    if inflight.exists():
        if read(inflight)['id'] not in committed:
            raise ValueError('Uncommitted sample: DO NOT reset, delete or rerun it. Preserve artifacts and investigate.')
        inflight.unlink()
    return committed
