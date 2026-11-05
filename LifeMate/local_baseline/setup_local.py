"""Download pinned CUDA llama.cpp and extract the official archive without changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = "b11800"  # Published 2026-10-21, before the competition resource cutoff.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path, *, skip_apple: bool = False) -> int:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            name = PurePosixPath(item.filename)
            if skip_apple and any(p == "__MACOSX" or p.startswith("._") or p == ".DS_Store" for p in name.parts):
                continue
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or any(":" in p for p in name.parts):
                raise ValueError(f"Unsafe archive member: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink rejected: {item.filename}")
            target = destination.joinpath(*name.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive target escaped destination")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # Preserve existing files; only reuse identical contents.
                with source.open(item) as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                if sha256(target) != expected:
                    raise FileExistsError(f"Existing extracted file differs: {target}")
            else:
                partial = target.with_name(target.name + ".extracting")
                with source.open(item) as fin, partial.open("xb") as fout:
                    shutil.copyfileobj(fin, fout)
                if partial.stat().st_size != item.file_size:
                    raise ValueError(f"Incomplete extraction: {target}")
                partial.rename(target)
            count += 1
    return count


def install_runtime() -> dict:
    runtime = ROOT / ".runtime" / TAG
    marker = runtime / "installation.json"
    if marker.exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if Path(record["server"]).is_file():
            print("RUNTIME_ALREADY_READY", record["server"], flush=True)
            return record
        raise RuntimeError("Runtime marker exists but server is missing")
    request = urllib.request.Request(
        f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}",
        headers={"User-Agent": "LifeMate-local-setup"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        release = json.load(response)
    cache = ROOT / ".runtime" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    assets = []
    for name in [f"llama-{TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]:
        asset = next(a for a in release["assets"] if a["name"] == name)
        target = cache / name
        expected = asset.get("digest", "").removeprefix("sha256:")
        if not expected:
            raise RuntimeError("Release asset has no SHA-256")
        if not target.exists():
            partial = target.with_suffix(".zip.part")
            print("DOWNLOADING", name, asset["size"], flush=True)
            subprocess.run([
                "curl.exe", "--location", "--fail", "--retry", "4", "--silent", "--show-error",
                "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial),
                asset["browser_download_url"],
            ], check=True)
            if partial.stat().st_size != asset["size"] or sha256(partial) != expected:
                raise RuntimeError(f"Downloaded asset failed verification: {name}")
            partial.rename(target)
        elif target.stat().st_size != asset["size"] or sha256(target) != expected:
            raise RuntimeError(f"Cached asset failed verification: {name}")
        print("EXTRACTING_RUNTIME", name, flush=True)
        safe_extract(target, runtime)
        assets.append({"name": name, "sha256": expected, "url": asset["browser_download_url"]})
    servers = list(runtime.rglob("llama-server.exe"))
    if len(servers) != 1:
        raise RuntimeError(f"Expected one llama-server.exe, found {servers}")
    # Different release archives may wrap their files in a directory.
    server = servers[0]
    for dll in runtime.rglob("*.dll"):
        destination = server.parent / dll.name
        if dll.parent != server.parent and not destination.exists():
            shutil.copy2(dll, destination)
    record = {"tag": TAG, "published_at": release["published_at"], "server": str(server), "assets": assets}
    marker.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("RUNTIME_READY", server, flush=True)
    return record


def extract_dataset() -> Path:
    archive = ROOT / "train+public_test.zip"
    destination = ROOT / "data" / "lifemate-v3.2.0"
    marker = destination / "extraction.json"
    if marker.exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if sha256(archive) != record["archive_sha256"]:
            raise RuntimeError("Source archive changed since extraction")
        return destination / "train+public_test"
    print("EXTRACTING_OFFICIAL_DATASET", flush=True)
    before = sha256(archive)
    count = safe_extract(archive, destination, skip_apple=True)
    if sha256(archive) != before:
        raise RuntimeError("Source archive changed during extraction")
    root = destination / "train+public_test"
    if not (root / "train" / "train.jsonl").is_file():
        raise RuntimeError("Unexpected dataset structure")
    marker.write_text(json.dumps({"archive_sha256": before, "files": count, "root": str(root)}, indent=2), encoding="utf-8")
    print("DATASET_READY", root, "files", count, flush=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if not args.data_only:
        install_runtime()
    extract_dataset()


if __name__ == "__main__":
    main()
