"""Self-contained release integrity; no original experiment paths or imports."""
from pathlib import Path
from .io import read, sha

HERE = Path(__file__).resolve().parents[1]

def verify_preservation():
    metadata = read(HERE / "SOURCE_PROVENANCE.json")
    for name, digest in metadata["inference_core_byte_identical_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Frozen baseline50.9 core changed: " + name)

def verify():
    verify_preservation()
    for name, digest in read(HERE / "CODE_MANIFEST.json")["files_sha256"].items():
        path = (HERE / name).resolve()
        if not path.is_relative_to(HERE) or not path.is_file() or sha(path) != digest:
            raise ValueError("Release file missing/changed: " + name)
