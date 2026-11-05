"""Two sequential live-ASR jobs on the frozen new 600 examples; local only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .expanded_selection import create, PATH as SELECTION
from .experiment import write_json
from .setup_local import ROOT, sha256


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    selection = create()
    directory = ROOT / "runs" / datetime.now(timezone.utc).strftime("expanded_campaign_%Y%m%dT%H%M%SZ")
    directory.mkdir(exist_ok=False)
    python = str(ROOT / ".runtime/asr-venv/Scripts/python.exe")
    jobs = [{"model": model, "mode": "asr", "status": "pending"} for model in ("qwen", "gemma")]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "jobs": jobs,
                "samples_per_job": len(selection["ids"]), "new_predictions": 1200,
                "selection_manifest": str(SELECTION), "selection_sha256": sha256(SELECTION),
                "future_training": selection["future_training"], "model_decision": "priority after reviewing results; no training"}
    write_json(directory / "campaign.json", manifest)
    print("EXPANDED_CAMPAIGN_DIR", directory, flush=True)
    for job in jobs:
        weights = ROOT / ("MODEL_DOWNLOADS_2026-11-04.json" if job["model"] == "qwen" else "GEMMA_MODEL_DOWNLOAD_2026-11-04.json")
        command = [python, "-u", "-m", "local_baseline.experiment", "--model", job["model"], "--mode", "asr",
                   "--partition", "expanded", "--ids-manifest", str(SELECTION), "--weights-manifest", str(weights)]
        job.update(status="running", command=command)
        write_json(directory / "campaign.json", manifest)
        try:
            with (directory / f"{job['model']}_asr.log").open("x", encoding="utf-8") as log:
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
                    code = process.wait()
            if code:
                raise RuntimeError(f"{job['model']} failed with exit {code}; artifacts preserved")
            metrics = json.loads((Path(job["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
            if metrics["samples"] != 600:
                raise RuntimeError("Incomplete expansion")
            job.update(status="completed", metrics=metrics)
            write_json(directory / "campaign.json", manifest)
        except BaseException as error:
            job.update(status="failed", error=repr(error))
            manifest["status"] = "failed"
            write_json(directory / "campaign.json", manifest)
            raise
    command = [python, "-u", "-m", "local_baseline.expanded_compare", "--qwen", jobs[0]["run_dir"],
               "--gemma", jobs[1]["run_dir"], "--output", str(directory / "comparison")]
    subprocess.run(command, cwd=ROOT, check=True)
    manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    report=str(directory / "comparison/REPORT.md"))
    write_json(directory / "campaign.json", manifest)
    print("EXPANDED_CAMPAIGN_COMPLETE", manifest["report"], flush=True)


if __name__ == "__main__":
    main()
