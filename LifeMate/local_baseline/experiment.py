"""Frozen full validation comparison, gold text or LIVE ASR, with official-like scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

from .agent import infer, SYSTEM
from .backend import LocalServer, MODELS
from .data import DATA_ROOT, MANIFEST, agent_view, catalog, prepare, read_rows
from .references import PATH as REFERENCE_PATH
from .scoring import ASSUMPTIONS, VERSION, score
from .setup_local import ROOT, TAG, sha256
from .speech import audio_paths, audio_view, normalize_transcript


def write_json(path, value):
    # Windows viewers/antivirus can briefly deny replacing an open file.
    # Use a unique staging file and bounded retry; preserve it if the lock persists.
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    until = time.monotonic() + 15
    while True:
        try:
            temp.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= until:
                raise
            time.sleep(0.1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=["gold", "asr"], required=True)
    parser.add_argument("--partition", choices=["val-main", "smoke", "expanded", "train-extra", "all-train"], default="val-main")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights-manifest", type=Path, default=ROOT / "MODEL_DOWNLOADS_2026-11-04.json")
    parser.add_argument("--ids-manifest", type=Path)
    args = parser.parse_args()
    split = prepare()
    ids = split["val_main"] if args.partition == "val-main" else split["smoke_train_fit"]
    if args.partition == "expanded":
        if args.ids_manifest is None or args.limit is not None:
            parser.error("expanded requires --ids-manifest and the complete frozen selection (no --limit)")
        from .expanded_selection import validate_selection
        ids = validate_selection(json.loads(args.ids_manifest.read_text(encoding="utf-8")), split)
    elif args.partition in ("train-extra", "all-train"):
        if args.limit is not None:
            parser.error("Full evaluation cannot be truncated with --limit")
        from .full_selection import select, create
        create()
        ids = select(args.partition, split)
    elif args.ids_manifest is not None:
        parser.error("--ids-manifest is only valid with --partition expanded")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        ids = ids[:args.limit]
    rows = {r["id"]: r for r in read_rows()}
    tools = catalog()
    critical = ["agent.py", "backend.py", "data.py", "scoring.py", "speech.py", "experiment.py"]
    if args.partition == "expanded":
        critical.append("expanded_selection.py")
    if args.partition in ("train-extra", "all-train"):
        critical.append("full_selection.py")
    code = {name: sha256(Path(__file__).parent / name) for name in critical}
    weights = json.loads(args.weights_manifest.read_text(encoding="utf-8"))["models"]
    weight = next(w for w in weights if w["filename"] == MODELS[args.model])
    if sha256(ROOT / weight["filename"]) != weight["expected_sha256"]:
        raise RuntimeError("Weights changed")
    common = {"model": args.model, "mode": args.mode, "partition": args.partition, "ids": ids,
              "split_sha256": sha256(MANIFEST), "catalog_sha256": sha256(DATA_ROOT / "train/train_tools_catalog.json"),
              "runtime_py_sha256": {str(p.relative_to(DATA_ROOT / "train/zaic_tools")): sha256(p)
                                    for p in sorted((DATA_ROOT / "train/zaic_tools").rglob("*.py"))},
              "reference_sha256": sha256(REFERENCE_PATH), "code_sha256": code,
              "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), "scorer": VERSION,
              "scorer_assumptions": ASSUMPTIONS, "runtime_tag": TAG, "seed": 20261103,
              "thinking": False, "temperature": 0, "context": 16384, "max_new_tokens": 512,
              "max_calls": 8, "max_repairs": 2, "deadline_s": 30, "fine_tuned": False, "weight": weight,
              "normalization": "NFC and whitespace only, gold and ASR",
              "timing": "live per sample, includes view/env/tokenization/ASR(if enabled)/LLM/tool execution; startup recorded separately",
              "not_official_score": True, "hardware_limit": "Local RTX4060, not A30 certification"}
    if args.partition == "expanded":
        common["selection_manifest_sha256"] = sha256(args.ids_manifest)
    if args.partition in ("train-extra", "all-train"):
        from .full_selection import PATH as FULL_SELECTION_PATH
        common["full_selection_manifest_sha256"] = sha256(FULL_SELECTION_PATH)
    run_dir = args.resume.resolve() if args.resume else ROOT / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_full_{args.model}_{args.mode}_" + uuid.uuid4().hex[:6])
    if args.resume:
        previous = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        for key, value in common.items():
            if previous.get(key) != value:
                raise RuntimeError("Cannot resume: configuration differs: " + key)
        manifest = previous
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {**common, "status": "preparing", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "startup_sessions": [], "completed_samples": 0}
        write_json(run_dir / "run_manifest.json", manifest)
        snapshot = run_dir / "code_snapshot"
        snapshot.mkdir()
        for name in critical:
            (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
    print("EXPERIMENT_DIR", run_dir, flush=True)
    if not args.resume:
        try:
            hardware = subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8", errors="replace")
            (run_dir / "hardware.txt").write_text(hardware, encoding="utf-8")
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    results = []
    trace_path = run_dir / "traces.jsonl"
    if trace_path.exists():
        lines = trace_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            # Only complete committed records can be resumed. Never silently discard a broken tail.
            results.append(json.loads(line))
        if [r["id"] for r in results] != ids[:len(results)]:
            raise RuntimeError("Resume records are not an exact prefix of the frozen sample list")
    startup = time.monotonic()
    asr = None
    if args.mode == "asr":
        from .speech import PhoWhisper
        asr = PhoWhisper()
        if manifest.get("asr") and manifest["asr"] != asr.config:
            raise RuntimeError("ASR provider changed on resume")
        manifest["asr"] = asr.config
        print("ASR_READY", json.dumps(asr.config), flush=True)
    else:
        manifest["asr"] = None
    if "zaic_tools" in sys.modules:
        raise RuntimeError("Fresh process required for split-specific runtime")
    sys.path.insert(0, str(DATA_ROOT / "train"))
    from zaic_tools import ToolEnv
    session_dir = run_dir / ("server-session-" + str(len(manifest["startup_sessions"]) + 1))
    with LocalServer(args.model, session_dir) as backend:
        manifest["startup_sessions"].append({"seconds": time.monotonic() - startup,
                                             "server_command": backend.public_command()})
        manifest["status"] = "running"
        write_json(run_dir / "run_manifest.json", manifest)
        with trace_path.open("a", encoding="utf-8") as output:
            for index in range(len(results), len(ids)):
                sample_id = ids[index]
                started = time.monotonic()
                env = ToolEnv("train", sample_id, root=DATA_ROOT)
                asr_details = None
                try:
                    if asr:
                        transcripts, asr_details = asr.transcribe(audio_paths(rows[sample_id]), started + 30)
                        view = audio_view(rows[sample_id], transcripts)
                    else:
                        view = agent_view(rows[sample_id])
                        view["latest_text"] = normalize_transcript(view["latest_text"])
                        for turn in view["history"]:
                            if turn["role"] == "user":
                                turn["text"] = normalize_transcript(turn["text"])
                    remaining = 30 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("ASR/input preparation exhausted end-to-end deadline")
                    result = infer(view, tools, env, backend, seconds=remaining)
                except TimeoutError as error:
                    result = {"id": sample_id, "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
                              "status": "timeout", "error": str(error), "trace": [], "repairs": 0}
                except Exception as error:
                    result = {"id": sample_id, "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
                              "status": "asr_error" if asr else "input_error", "error": repr(error), "trace": [], "repairs": 0}
                elapsed = time.monotonic() - started
                result["llm_and_agent_s"] = result.pop("elapsed_s", 0.0)
                result["elapsed_s"] = elapsed
                result["asr"] = asr_details
                if elapsed > 30:
                    result["status"] = "timeout"
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                results.append(result)
                manifest["completed_samples"] = len(results)
                write_json(run_dir / "run_manifest.json", manifest)
                print(f"[{index+1}/{len(ids)}] {sample_id} {result['status']} calls={len(result['tool_calls'])} total={elapsed:.2f}s asr={asr_details['elapsed_s'] if asr_details else 0:.2f}s", flush=True)
    # Scoring runs after inference, outside the live agent's inputs.
    references = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    reports = []
    for result in results:
        ref = references["references"][result["id"]]
        failed = result["status"] not in ("finished", "asked_missing_info")
        value = score(result, ref, tools, failed=failed)
        reports.append({"id": result["id"], "status": result["status"], "elapsed_s": result["elapsed_s"],
                        "llm_and_agent_s": result["llm_and_agent_s"], "asr_s": result["asr"]["elapsed_s"] if result["asr"] else 0,
                        "asr_transcript_sha256": result["asr"]["transcript_sha256"] if result["asr"] else None,
                        "partial_label_consistent": ref["partial_label_consistent"], **value})
    def percentiles(values):
        values = sorted(values)
        return {"p50_s": statistics.median(values), "p95_s": values[math.ceil(.95*len(values))-1], "max_s": max(values)}
    summary = {"model": args.model, "mode": args.mode, "samples": len(reports), "expected_samples": len(ids),
               "score_spec_local": statistics.mean(r["score_spec_local"] for r in reports),
               "tool_f1": statistics.mean(r["tool_f1"] for r in reports),
               "strict_tool_f1": statistics.mean(r["strict_tool_f1"] for r in reports),
               "state_accuracy": statistics.mean(r["state_match"] for r in reports),
               "statuses": dict(Counter(r["status"] for r in reports)),
               "latency": percentiles([r["elapsed_s"] for r in reports]),
               "asr_latency": percentiles([r["asr_s"] for r in reports]) if asr else None,
               "scorer": VERSION, "not_official_score": True,
               "failed_samples_counted_zero": sum(r["failure_override"] for r in reports)}
    with (run_dir / "submission.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps({k: result[k] for k in ("id", "tool_calls", "state_hash")}, ensure_ascii=False) + "\n")
    (run_dir / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reports), encoding="utf-8")
    write_json(run_dir / "metrics.json", summary)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "run_manifest.json", manifest)
    lines = [f"# {args.model} — {args.mode} — {args.partition}", "", "Điểm local bám công thức BTC; không phải điểm leaderboard.",
             f"Mẫu: {len(reports)}; score: {summary['score_spec_local']:.4f}/100; timing: {'live ASR + LLM + tools' if asr else 'gold-text LLM + tools'}.",
             "", "| ID | Trạng thái | Điểm | Hash đúng | Calls đúng/gold | Tổng giây | ASR giây |", "|---|---|---|---|---|---|---|"]
    for r in reports:
        lines.append(f"| {r['id']} | {r['status']} | {r['score_spec_local']:.2f} | {r['state_match']} | {r['matched_calls']}/{r['gold_calls']} | {r['elapsed_s']:.2f} | {r['asr_s']:.2f} |")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("EXPERIMENT_COMPLETE", run_dir, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
