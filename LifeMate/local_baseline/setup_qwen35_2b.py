"""Pinned download of the additional Qwen3.5-2B baseline."""
from datetime import datetime, timezone
import json
import time
import urllib.request
from .setup_local import ROOT, sha256

REPO = "unsloth/Qwen3.5-2B-GGUF"
REVISION = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
FILENAME = "Qwen3.5-2B-Q4_K_M.gguf"
SIZE = 1280835840
SHA = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"
MANIFEST = ROOT / "QWEN35_2B_MODEL_DOWNLOAD_2026-11-04.json"


def main():
    target = ROOT / FILENAME
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
    if not target.exists():
        temp = target.with_suffix(".gguf.part")
        if temp.exists():
            raise RuntimeError("Partial file exists; inspect before retry: " + str(temp))
        request = urllib.request.Request(url, headers={"User-Agent": "LifeMate-local-evaluation"})
        received = 0
        report = time.monotonic()
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("xb") as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                received += len(block)
                if time.monotonic() - report >= 20:
                    print(f"DOWNLOAD {received}/{SIZE} ({received/SIZE:.1%})", flush=True)
                    report = time.monotonic()
        if received != SIZE or sha256(temp) != SHA:
            raise RuntimeError("Size/hash mismatch; partial preserved")
        temp.rename(target)
    if target.stat().st_size != SIZE or sha256(target) != SHA:
        raise RuntimeError("Weights failed verification")
    record = {"repository": REPO, "revision": REVISION, "base_model": "Qwen/Qwen3.5-2B",
              "license": "apache-2.0", "filename": FILENAME, "format": "GGUF", "quantization": "Q4_K_M",
              "expected_bytes": SIZE, "expected_sha256": SHA, "source_url": url, "status": "verified",
              "actual_bytes": SIZE, "actual_sha256": SHA}
    if not MANIFEST.exists():
        with MANIFEST.open("x", encoding="utf-8") as stream:
            json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat(), "models": [record],
                       "purpose": "Additional full-train zero-shot baseline; local only"}, stream, ensure_ascii=False, indent=2)
    elif json.loads(MANIFEST.read_text(encoding="utf-8"))["models"] != [record]:
        raise RuntimeError("Manifest differs; not overwritten")
    print("QWEN35_2B_VERIFIED", target, flush=True)


if __name__ == "__main__":
    main()
