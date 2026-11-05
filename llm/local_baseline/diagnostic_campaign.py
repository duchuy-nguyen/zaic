"""Full Qwen3.5-4B clean-text diagnosis; reuses frozen full audio predictions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .experiment import write_json
from .full_selection import create, PATH as SELECTION
from .setup_local import ROOT, sha256


ASR_RUNS = [ROOT / "runs/20261104T143403Z_full_qwen35_asr_688fa7",
            ROOT / "runs/20261104T210559Z_full_qwen35_asr_88383a"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Existing campaign DIRECTORY")
    args = parser.parse_args()
    selection = create()
    asr_ids = []
    for path in ASR_RUNS:
        saved = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        if saved["status"] != "completed" or saved["model"] != "qwen35" or saved["mode"] != "asr":
            raise ValueError("Frozen audio baseline is incomplete or has changed")
        asr_ids.extend(saved["ids"])
    if len(asr_ids) != 1545 or sorted(asr_ids) != selection["all_train"]:
        raise ValueError("Audio baseline must cover every ID exactly once")
    base = json.loads((ASR_RUNS[1] / "run_manifest.json").read_text(encoding="utf-8"))
    for name, expected in base["code_sha256"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError("Inference source changed since the audio run: " + name)
    if args.resume:
        directory = args.resume.resolve()
        manifest = json.loads((directory / "campaign.json").read_text(encoding="utf-8"))
        if manifest["selection_sha256"] != sha256(SELECTION) or manifest["asr_runs"] != list(map(str, ASR_RUNS)):
            raise ValueError("Frozen campaign inputs changed")
        if manifest["status"] == "completed":
            print("DIAGNOSTIC_ALREADY_COMPLETE", manifest["report"], flush=True)
            return
    else:
        directory = ROOT / "runs" / datetime.now(timezone.utc).strftime("diagnostic_qwen35_campaign_%Y%m%dT%H%M%SZ")
        directory.mkdir(exist_ok=False)
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
                    "selection_sha256": sha256(SELECTION), "asr_runs": list(map(str, ASR_RUNS)),
                    "expected_samples": 1545, "sessions": 0,
                    "scope": "Local zero-shot clean-text vs existing live-ASR; no fine-tune or upload"}
    # OS lock is released on exit/crash; a stale file never blocks checkpoint resume.
    lock = (directory / "coordinator.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    python = str(ROOT / ".runtime/asr-venv/Scripts/python.exe")
    manifest.update(status="running")
    write_json(directory / "campaign.json", manifest)
    print("DIAGNOSTIC_CAMPAIGN_DIR", directory, flush=True)
    try:
        existing = Path(manifest["run_dir"]) if manifest.get("run_dir") else None
        done = existing and (existing / "metrics.json").is_file() and json.loads(
            (existing / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "completed"
        if not done:
            command = [python, "-u", "-m", "local_baseline.experiment", "--model", "qwen35", "--mode", "gold",
                       "--partition", "all-train", "--weights-manifest", str(ROOT / "QWEN35_MODEL_DOWNLOAD_2026-11-04.json")]
            if existing:
                command += ["--resume", str(existing)]
            manifest.update(sessions=manifest["sessions"]+1, command=command)
            write_json(directory / "campaign.json", manifest)
            with (directory / f"gold_session{manifest['sessions']}.log").open("x", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
                raise RuntimeError(f"Gold experiment exited {code}; resume checkpoints without rerunning committed samples")
        metrics = json.loads((Path(manifest["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        if metrics["samples"] != 1545:
            raise ValueError("Incomplete full gold-text run")
        manifest.update(status="analyzing", metrics=metrics)
        write_json(directory / "campaign.json", manifest)
        # A previous interrupted analysis is preserved; create a new output directory.
        output = directory / "comparison"
        if output.exists():
            output = directory / datetime.now(timezone.utc).strftime("comparison_%Y%m%dT%H%M%SZ")
        subprocess.run([python, "-u", "-m", "local_baseline.diagnostic_compare", "--gold", manifest["run_dir"],
                        "--output", str(output)], cwd=Path(__file__).resolve().parents[1], check=True)
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        report=str(output / "REPORT.md"), csv=str(output / "metrics.csv"),
                        cases=str(output / "ERROR_CASES.md"), analysis=str(output / "analysis.json"))
        manifest.pop("error", None)
        write_json(directory / "campaign.json", manifest)
        print("DIAGNOSTIC_CAMPAIGN_COMPLETE", manifest["report"], flush=True)
    except BaseException as error:
        manifest.update(status="interrupted_or_failed", error=repr(error))
        write_json(directory / "campaign.json", manifest)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
