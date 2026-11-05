"""Download one pinned Gemma quantization, without changing earlier manifests."""
from datetime import datetime, timezone
import json
import time
import urllib.request

from .setup_local import ROOT, sha256

REPO = "unsloth/gemma-4-E2B-it-GGUF"
REVISION = "0314792d7f1f7e229411f620751375812bb9faf2"
FILENAME = "gemma-4-E2B-it-Q4_K_M.gguf"
SIZE = 3106738272
SHA = "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8"
MANIFEST = ROOT / "GEMMA_MODEL_DOWNLOAD_2026-11-04.json"


def main():
    target = ROOT / FILENAME
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
    if not target.exists():
        temp = target.with_suffix(".gguf.part")
        if temp.exists():
            raise RuntimeError(f"Partial download exists; inspect before retry: {temp}")
        request = urllib.request.Request(url, headers={"User-Agent": "LifeMate-local-evaluation"})
        started = report = time.monotonic()
        received = 0
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("xb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                received += len(block)
                if time.monotonic() - report >= 20:
                    print(f"DOWNLOAD {received}/{SIZE} bytes ({received/SIZE:.1%}) {received/(time.monotonic()-started)/1e6:.1f} MB/s", flush=True)
                    report = time.monotonic()
        if received != SIZE or sha256(temp) != SHA:
            raise RuntimeError("Downloaded model failed size/SHA256 verification; partial preserved")
        temp.rename(target)
    if target.stat().st_size != SIZE or sha256(target) != SHA:
        raise RuntimeError("Existing weights failed verification")
    record = {"repository": REPO, "revision": REVISION, "base_model": "google/gemma-4-E2B-it",
              "license": "apache-2.0", "filename": FILENAME, "format": "GGUF", "quantization": "Q4_K_M",
              "expected_bytes": SIZE, "expected_sha256": SHA, "source_url": url, "status": "verified",
              "actual_bytes": SIZE, "actual_sha256": SHA}
    if not MANIFEST.exists():
        with MANIFEST.open("x", encoding="utf-8") as output:
            json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat(), "models": [record],
                       "purpose": "Additional local zero-shot baseline; original Qwen/Spark manifest unchanged"}, output, ensure_ascii=False, indent=2)
    else:
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing["models"] != [record]:
            raise RuntimeError("Existing manifest differs; not overwriting")
    print("GEMMA_VERIFIED", target, flush=True)


if __name__ == "__main__":
    main()
