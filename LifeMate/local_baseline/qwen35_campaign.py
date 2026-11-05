"""Run Qwen3.5's 309 live-audio examples, then produce controlled comparison."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .experiment import write_json
from .setup_local import ROOT


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    directory = ROOT / "runs" / datetime.now(timezone.utc).strftime("qwen35_campaign_%Y%m%dT%H%M%SZ")
    directory.mkdir(exist_ok=False)
    python = str(ROOT / ".runtime/asr-venv/Scripts/python.exe")
    command = [python, "-u", "-m", "local_baseline.experiment", "--model", "qwen35", "--mode", "asr",
               "--partition", "val-main", "--weights-manifest", str(ROOT / "QWEN35_MODEL_DOWNLOAD_2026-11-04.json")]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "samples": 309, "model": "qwen35", "command": command,
                "next_stage": "Review 309 evidence before choosing whether full 1545 is needed"}
    write_json(directory / "campaign.json", manifest)
    print("QWEN35_CAMPAIGN_DIR", directory, flush=True)
    try:
        with (directory / "qwen35_asr.log").open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    if line.startswith("EXPERIMENT_DIR "):
                        manifest["run_dir"] = line[len("EXPERIMENT_DIR "):].strip()
                        write_json(directory / "campaign.json", manifest)
                code = process.wait()
        if code:
            raise RuntimeError(f"Qwen3.5 failed with exit {code}; artifacts preserved")
        metrics = json.loads((Path(manifest["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        if metrics["samples"] != 309:
            raise RuntimeError("Incomplete Qwen3.5 validation")
        manifest.update(status="analyzing", metrics=metrics)
        write_json(directory / "campaign.json", manifest)
        subprocess.run([python, "-u", "-m", "local_baseline.qwen35_compare", "--qwen35", manifest["run_dir"],
                        "--output", str(directory / "comparison")], cwd=ROOT, check=True)
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        report=str(directory / "comparison/REPORT.md"), csv=str(directory / "comparison/metrics.csv"))
        write_json(directory / "campaign.json", manifest)
        print("QWEN35_CAMPAIGN_COMPLETE", manifest["report"], flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        write_json(directory / "campaign.json", manifest)
        raise


if __name__ == "__main__":
    main()
