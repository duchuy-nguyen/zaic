"""Safely unpack a shared Drive archive without overwriting different files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import uuid
import zipfile


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def unpack(archive, destination, checksums=None):
    archive = Path(archive).resolve(); destination = Path(destination).resolve()
    if checksums:
        expected = json.loads(Path(checksums).read_text(encoding='utf8'))[archive.name]
        if archive.stat().st_size != expected['bytes'] or sha(archive) != expected['sha256']:
            raise ValueError('Archive SHA256/size mismatch')
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        files = json.loads(z.read('ARCHIVE_MANIFEST.json'))['files']
        if set(z.namelist()) != set(files) | {'ARCHIVE_MANIFEST.json'}:
            raise ValueError('Unexpected ZIP members')
        if sum(v['bytes'] for v in files.values()) > 16 * 1024**3:
            raise ValueError('Archive unexpectedly large')
        for name, info in files.items():
            rel = PurePosixPath(name); member = z.getinfo(name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe ZIP path: ' + name)
            target = destination.joinpath(*rel.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('ZIP path escapes destination')
            if target.exists():
                if target.stat().st_size != info['bytes'] or sha(target) != info['sha256']:
                    raise ValueError('Preserve different existing file: ' + str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.' + uuid.uuid4().hex + '.part')
            with z.open(name) as source, partial.open('xb') as out:
                shutil.copyfileobj(source, out, length=8 * 1024**2)
            if partial.stat().st_size != info['bytes'] or sha(partial) != info['sha256']:
                raise ValueError('Extracted member failed SHA256; preserve partial for inspection')
            partial.rename(target)
    # Keep each manifest separately; multiple ZIPs never overwrite one another's manifest.
    manifests = destination / '.archive_manifests'; manifests.mkdir(exist_ok=True)
    target = manifests / (archive.name + '.json')
    payload = json.dumps(files, indent=2) + '\n'
    if target.exists():
        if target.read_text(encoding='utf8') != payload: raise ValueError('Existing archive manifest differs')
    else:
        with target.open('x', encoding='utf8') as f: f.write(payload)
    print('UNPACKED_AND_VERIFIED', archive.name)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--dest', type=Path, default=Path('private_assets'))
    p.add_argument('--checksums', type=Path)
    args = p.parse_args(); unpack(args.archive, args.dest, args.checksums)
