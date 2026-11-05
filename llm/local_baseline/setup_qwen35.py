"""Download pinned Qwen3.5-4B Q4_K_M weights; preserve earlier model manifests."""
from datetime import datetime, timezone
import json
import time
import urllib.request

from .setup_local import ROOT, sha256

REPO = "unsloth/Qwen3.5-4B-GGUF"
REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
FILENAME = "Qwen3.5-4B-Q4_K_M.gguf"
SIZE = 2740937888
SHA = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
MANIFEST = ROOT / "QWEN35_MODEL_DOWNLOAD_2026-11-04.json"


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
            raise RuntimeError("Downloaded weights failed size/SHA256 verification; partial preserved")
        temp.rename(target)
    if target.stat().st_size != SIZE or sha256(target) != SHA:
        raise RuntimeError("Existing weights failed verification")
    record = {"repository": REPO, "revision": REVISION, "base_model": "Qwen/Qwen3.5-4B",
              "license": "apache-2.0", "filename": FILENAME, "format": "GGUF", "quantization": "Q4_K_M",
              "expected_bytes": SIZE, "expected_sha256": SHA, "source_url": url, "status": "verified",
              "actual_bytes": SIZE, "actual_sha256": SHA}
    if not MANIFEST.exists():
        with MANIFEST.open("x", encoding="utf-8") as output:
            json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat(), "models": [record],
                       "purpose": "Qwen3.5 local zero-shot comparison; no training or uploads"}, output, ensure_ascii=False, indent=2)
    elif json.loads(MANIFEST.read_text(encoding="utf-8"))["models"] != [record]:
        raise RuntimeError("Existing manifest differs; not overwriting")
    print("QWEN35_VERIFIED", target, flush=True)


if __name__ == "__main__":
    main()
