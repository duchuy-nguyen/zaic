"""Resumable sequential full-train evaluation for three requested models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .experiment import write_json
from .full_selection import create, PATH as SELECTION
from .setup_local import ROOT, sha256


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    create()
    if args.resume:
        directory = args.resume.resolve()
        manifest = json.loads((directory / "campaign.json").read_text(encoding="utf-8"))
        if manifest["selection_sha256"] != sha256(SELECTION):
            raise RuntimeError("Selection changed")
        if manifest["status"] == "completed":
            print("FULL_CAMPAIGN_ALREADY_COMPLETE", manifest["report"], flush=True)
            return
    else:
        directory = ROOT / "runs" / datetime.now(timezone.utc).strftime("full_three_campaign_%Y%m%dT%H%M%SZ")
        directory.mkdir(exist_ok=False)
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                    "selection_sha256": sha256(SELECTION), "new_predictions": 4017, "total_predictions": 4635,
                    "jobs": [{"model": model, "partition": partition, "expected_samples": n, "status": "pending", "sessions": 0}
                             for model, partition, n in [("qwen35_2b", "all-train", 1545), ("gemma", "train-extra", 1236), ("qwen35", "train-extra", 1236)]]}
    manifest["status"] = "running"
    write_json(directory / "campaign.json", manifest)
    print("FULL_CAMPAIGN_DIR", directory, flush=True)
    python = str(ROOT / ".runtime/asr-venv/Scripts/python.exe")
    weights = {"qwen35_2b": "QWEN35_2B_MODEL_DOWNLOAD_2026-11-04.json",
               "qwen35": "QWEN35_MODEL_DOWNLOAD_2026-11-04.json", "gemma": "GEMMA_MODEL_DOWNLOAD_2026-11-04.json"}
    try:
        for job in manifest["jobs"]:
            if job["status"] == "completed":
                continue
            existing = Path(job["run_dir"]) if job.get("run_dir") else None
            if existing and (existing / "metrics.json").is_file() and json.loads((existing / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "completed":
                metrics = json.loads((existing / "metrics.json").read_text(encoding="utf-8"))
            else:
                command = [python, "-u", "-m", "local_baseline.experiment", "--model", job["model"], "--mode", "asr",
                           "--partition", job["partition"], "--weights-manifest", str(ROOT / weights[job["model"]])]
                if existing:
                    command += ["--resume", str(existing)]
                job.update(status="running", command=command, sessions=job["sessions"]+1)
                write_json(directory / "campaign.json", manifest)
                print("START_FULL_JOB", job["model"], job["expected_samples"], flush=True)
                with (directory / f"{job['model']}_session{job['sessions']}.log").open("x", encoding="utf-8") as log:
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
                    raise RuntimeError(f"{job['model']} exit {code}; preserve artifacts and use --resume")
                metrics = json.loads((Path(job["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            if metrics["samples"] != job["expected_samples"]:
                raise RuntimeError("Incomplete full job")
            job.update(status="completed", metrics=metrics)
            write_json(directory / "campaign.json", manifest)
        manifest["status"] = "analyzing"
        write_json(directory / "campaign.json", manifest)
        command = [python, "-u", "-m", "local_baseline.full_compare"]
        for job in manifest["jobs"]:
            command += ["--"+job["model"], job["run_dir"]]
        command += ["--output", str(directory / "comparison")]
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        report=str(directory / "comparison/REPORT.md"), csv=str(directory / "comparison/metrics.csv"))
        write_json(directory / "campaign.json", manifest)
        print("FULL_CAMPAIGN_COMPLETE", manifest["report"], flush=True)
    except BaseException as error:
        manifest.update(status="interrupted_or_failed", error=repr(error))
        write_json(directory / "campaign.json", manifest)
        raise


if __name__ == "__main__":
    main()
