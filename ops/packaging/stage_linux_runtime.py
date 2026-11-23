"""Materialize a separately pinned Linux candidate; never changes Windows pins."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ARCHIVE_SHA256='e5d489b995c2a0965ee746a246c934be9f4ffa00c78183b2d4f644f43ae5d7a7'
ARCHIVE_BYTES=177920437
SOURCE='https://github.com/ggml-org/llama.cpp/releases/download/b11800/llama-b11800-bin-ubuntu-cuda-12.8-x64.tar.gz'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def stage(archive,destination):
    archive=Path(archive);destination=Path(destination)
    if destination.exists():raise FileExistsError('Preserve existing runtime stage')
    if archive.stat().st_size!=ARCHIVE_BYTES or sha(archive)!=ARCHIVE_SHA256:
        raise ValueError('Upstream Linux archive differs from pinned release asset')
    with tarfile.open(archive,'r:gz') as tar:
        members=[m for m in tar.getmembers() if not m.isdir()]
        if len({m.name for m in members})!=len(members):raise ValueError('Duplicate archive members')
        for member in members:
            path=PurePosixPath(member.name)
            if len(path.parts)!=2 or path.parts[0]!='llama-b11800' or '..' in path.parts:
                raise ValueError('Unexpected runtime layout')
            if not (member.isfile() or member.issym()) or (member.issym() and ('/' in member.linkname or '\\' in member.linkname)):
                raise ValueError('Unexpected runtime member/link')
        destination.mkdir(parents=True,exist_ok=False);files={}
        for member in members:
            target=destination/PurePosixPath(member.name).name
            # Flatten only this immutable release, dereference its local aliases.
            with tar.extractfile(member) as source,target.open('xb') as output:
                shutil.copyfileobj(source,output)
            target.chmod(0o755 if 'llama' in target.name and '.so' not in target.name else 0o644)
            files[target.name]={'bytes':target.stat().st_size,'sha256':sha(target)}
    receipt={'release':'b11800','platform':'linux-amd64-cuda12.8','server':'llama-server',
             'source':SOURCE,'archive_sha256':ARCHIVE_SHA256,'parity_verified':False,'files':files}
    (destination/'runtime_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();r=stage(a.archive,a.destination);print(json.dumps({'files':len(r['files']),'parity_verified':False}))
