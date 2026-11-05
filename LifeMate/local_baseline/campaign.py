"""Execute both text baselines then both live-audio baselines sequentially on one GPU."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .setup_local import ROOT
from .experiment import write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    asr_python = ROOT / ".runtime/asr-venv/Scripts/python.exe"
    if not asr_python.is_file():
        raise RuntimeError("ASR environment not prepared")
    directory = ROOT / "runs" / (datetime.now(timezone.utc).strftime("validation_campaign_%Y%m%dT%H%M%SZ"))
    directory.mkdir(exist_ok=False)
    jobs = [{"model": model, "mode": mode, "status": "pending"}
            for mode in ("gold", "asr") for model in ("qwen", "spark")]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "jobs": jobs,
                "samples_per_job": 309, "total_predictions": 1236, "model_decision": "deferred_to_user"}
    write_json(directory / "campaign.json", manifest)
    print("CAMPAIGN_DIR", directory, flush=True)
    for job in jobs:
        python = str(asr_python) if job["mode"] == "asr" else sys.executable
        command = [python, "-u", "-m", "local_baseline.experiment", "--model", job["model"],
                   "--mode", job["mode"], "--partition", "val-main"]
        job["status"] = "running"
        job["command"] = command
        write_json(directory / "campaign.json", manifest)
        print("START_JOB", job["model"], job["mode"], flush=True)
        log_path = directory / f"{job['model']}_{job['mode']}.log"
        with log_path.open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace",
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    if line.startswith("EXPERIMENT_DIR "):
                        job["run_dir"] = line[len("EXPERIMENT_DIR "):].strip()
                        write_json(directory / "campaign.json", manifest)
                exit_code = process.wait()
        job["exit_code"] = exit_code
        if exit_code:
            job["status"] = "failed"
            manifest["status"] = "failed"
            write_json(directory / "campaign.json", manifest)
            raise SystemExit(f"Campaign stopped: {job['model']} {job['mode']} failed; see {log_path}")
        metrics = json.loads((Path(job["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        if metrics["samples"] != 309:
            raise RuntimeError("Incomplete full validation; cannot proceed as if complete")
        if job["mode"] == "asr" and metrics["statuses"].get("asr_error", 0) == 309:
            raise RuntimeError("All ASR samples failed; investigate provider")
        job["status"] = "completed"
        job["metrics"] = metrics
        write_json(directory / "campaign.json", manifest)
    command = [sys.executable, "-m", "local_baseline.analyze_experiments"]
    for job in jobs:
        command += [f"--{job['model']}-{job['mode']}", job["run_dir"]]
    command += ["--output", str(directory / "comparison")]
    subprocess.run(command, cwd=ROOT, check=True)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["report"] = str(directory / "comparison/REPORT.md")
    write_json(directory / "campaign.json", manifest)
    print("CAMPAIGN_COMPLETE", manifest["report"], flush=True)


if __name__ == "__main__":
    main()
