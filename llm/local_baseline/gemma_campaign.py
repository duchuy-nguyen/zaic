"""Run Gemma text/audio validation sequentially, then compare all three models."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .experiment import write_json
from .setup_local import ROOT


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    directory = ROOT / "runs" / datetime.now(timezone.utc).strftime("gemma_campaign_%Y%m%dT%H%M%SZ")
    directory.mkdir(exist_ok=False)
    jobs = [{"model": "gemma", "mode": mode, "status": "pending"} for mode in ("gold", "asr")]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "jobs": jobs,
                "samples_per_job": 309, "total_predictions": 618, "model_decision": "deferred_to_user"}
    write_json(directory / "campaign.json", manifest)
    print("GEMMA_CAMPAIGN_DIR", directory, flush=True)
    for job in jobs:
        python = str(ROOT / ".runtime/asr-venv/Scripts/python.exe") if job["mode"] == "asr" else sys.executable
        command = [python, "-u", "-m", "local_baseline.experiment", "--model", "gemma", "--mode", job["mode"],
                   "--partition", "val-main", "--weights-manifest", str(ROOT / "GEMMA_MODEL_DOWNLOAD_2026-11-04.json")]
        job.update(status="running", command=command)
        write_json(directory / "campaign.json", manifest)
        try:
            with (directory / f"gemma_{job['mode']}.log").open("x", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace",
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                        if line.startswith("EXPERIMENT_DIR "):
                            job["run_dir"] = line[len("EXPERIMENT_DIR "):].strip()
                            write_json(directory / "campaign.json", manifest)
                    code = process.wait()
            if code:
                raise RuntimeError(f"{job['mode']} failed with exit {code}; artifacts preserved")
            metrics = json.loads((Path(job["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            if metrics["samples"] != 309:
                raise RuntimeError("Incomplete validation")
            job.update(status="completed", metrics=metrics)
            write_json(directory / "campaign.json", manifest)
        except BaseException as error:
            job.update(status="failed", error=repr(error))
            manifest["status"] = "failed"
            write_json(directory / "campaign.json", manifest)
            raise
    command = [sys.executable, "-u", "-m", "local_baseline.compare_three",
               "--gemma-gold", jobs[0]["run_dir"], "--gemma-asr", jobs[1]["run_dir"],
               "--output", str(directory / "comparison")]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    report=str(directory / "comparison/REPORT.md"))
    write_json(directory / "campaign.json", manifest)
    print("GEMMA_CAMPAIGN_COMPLETE", manifest["report"], flush=True)


if __name__ == "__main__":
    main()
