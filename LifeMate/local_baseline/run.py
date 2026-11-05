"""Run Qwen or Spark on TRAIN gold text with live official tools; no ASR or training."""
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

from .agent import SYSTEM, infer
from .backend import LocalServer, MODELS
from .data import DATA_ROOT, MANIFEST, agent_view, catalog, prepare, read_rows
from .references import PATH as REFERENCE_PATH
from .scoring import ASSUMPTIONS, VERSION, score
from .setup_local import ROOT, TAG, sha256


def percentile(values: list[float], p: float) -> float:
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)] if values else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="qwen")
    parser.add_argument("--partition", choices=["smoke", "val-main"], default="smoke")
    parser.add_argument("--limit", type=int, help="Explicit subset only; never described as the full partition")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--hello-only", action="store_true", help="Check model loading and Vietnamese generation only")
    args = parser.parse_args()
    if args.seconds <= 0 or args.seconds > 30:
        parser.error("Per-sample deadline must be >0 and <=30 seconds")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    sys.stdout.reconfigure(encoding="utf-8")
    split = prepare()
    if not REFERENCE_PATH.exists() and not args.hello_only:
        raise SystemExit("First run: python -m local_baseline.references (separate TRAIN annotation job)")
    ids = split["smoke_train_fit"] if args.partition == "smoke" else split["val_main"]
    if args.limit:
        ids = ids[:args.limit]
    rows = {row["id"]: row for row in read_rows()}
    tools = catalog()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{args.model}_{args.partition}_" + uuid.uuid4().hex[:6]
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    code_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    downloads = json.loads((ROOT / "MODEL_DOWNLOADS_2026-11-04.json").read_text(encoding="utf-8"))
    weight = next(m for m in downloads["models"] if m["filename"] == MODELS[args.model])
    if sha256(ROOT / weight["filename"]) != weight["expected_sha256"]:
        raise RuntimeError("Model SHA-256 differs from download manifest")
    manifest = {"run_id": run_id, "model": args.model, "weight": weight, "runtime_tag": TAG,
                "partition": args.partition, "ids": ids, "partition_total": len(split["smoke_train_fit"] if args.partition == "smoke" else split["val_main"]),
                "split_sha256": sha256(MANIFEST), "scorer": VERSION, "scorer_assumptions": ASSUMPTIONS,
                "code_sha256": code_hashes, "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "mode": "gold_current_and_history_text", "asr": None, "fine_tuned": False,
                "context": args.context, "max_new_tokens": 512, "max_calls": 8, "max_repairs": 2,
                "seconds": args.seconds, "seed": 20261103, "temperature": 0, "thinking": False,
                "limitations": ["Local Windows RTX4060 text-only smoke is not A30/audio timing certification",
                                "Smoke uses TRAIN_FIT, not held-out validation; no model ranking from smoke",
                                "Spec-aligned local scorer is not an official certified evaluator"]}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_DIR", run_dir, flush=True)
    try:
        gpu = subprocess.check_output(["nvidia-smi"], text=True, encoding="utf-8", errors="replace")
        (run_dir / "hardware.txt").write_text(gpu, encoding="utf-8")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    predictions = []
    started = time.monotonic()
    with LocalServer(args.model, run_dir, context=args.context) as backend:
        manifest["server_command"] = backend.public_command()
        manifest["startup_seconds"] = time.monotonic() - started
        if args.hello_only:
            response, count = backend.generate(
                [{"role": "user", "content": "Trả lời ngắn bằng tiếng Việt: bạn đã sẵn sàng hỗ trợ chưa?"}], [],
                time.monotonic() + args.seconds, max_tokens=64)
            (run_dir / "hello.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
            print("HELLO", response["choices"][0]["message"].get("content"), flush=True)
        else:
            if "zaic_tools" in sys.modules:
                raise RuntimeError("Runtime already loaded; use a fresh process per split")
            sys.path.insert(0, str(DATA_ROOT / "train"))
            from zaic_tools import ToolEnv
            with (run_dir / "submission.jsonl").open("x", encoding="utf-8") as output, (run_dir / "traces.jsonl").open("x", encoding="utf-8") as traces:
                for index, sample_id in enumerate(ids, 1):
                    sample_started = time.monotonic()
                    view = agent_view(rows[sample_id])
                    env = ToolEnv("train", sample_id, root=DATA_ROOT)  # Exactly one agent session per sample.
                    remaining = args.seconds - (time.monotonic() - sample_started)
                    result = infer(view, tools, env, backend, seconds=max(0.001, remaining))
                    result["elapsed_s"] = time.monotonic() - sample_started
                    if result["elapsed_s"] > args.seconds:
                        result["status"] = "timeout"
                    prediction = {k: result[k] for k in ("id", "tool_calls", "state_hash")}
                    output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    output.flush()
                    traces.write(json.dumps(result, ensure_ascii=False) + "\n")
                    traces.flush()
                    predictions.append(result)
                    print(f"[{index}/{len(ids)}] {sample_id} {result['status']} calls={len(result['tool_calls'])} time={result['elapsed_s']:.2f}s", flush=True)
    manifest["wall_seconds_including_startup"] = time.monotonic() - started
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.hello_only:
        print("HELLO_COMPLETE", run_dir, flush=True)
        return
    # Gold references are consulted only AFTER model execution is finished and server stopped.
    references = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    reports = []
    for result in predictions:
        ref = references["references"][result["id"]]
        failed = result["status"] not in ("finished", "asked_missing_info")
        metrics = score(result, ref, tools, failed=failed)
        reports.append({"id": result["id"], "status": result["status"], "elapsed_s": result["elapsed_s"],
                        "partial_label_consistent": ref["partial_label_consistent"], **metrics})
    times = [r["elapsed_s"] for r in reports]
    summary = {"samples": len(reports), "partition": args.partition, "partition_total": manifest["partition_total"],
               "score_spec_local": statistics.mean(r["score_spec_local"] for r in reports),
               "tool_f1": statistics.mean(r["tool_f1"] for r in reports),
               "state_accuracy": statistics.mean(r["state_match"] for r in reports),
               "statuses": dict(Counter(r["status"] for r in reports)),
               "latency_text_only": {"p50_s": statistics.median(times), "p95_s": percentile(times, .95), "max_s": max(times)},
               "reference_partial_mismatches_in_run": [r["id"] for r in reports if not r["partial_label_consistent"]],
               "scorer": VERSION, "not_official_score": True, "not_model_selection_evidence": args.partition == "smoke"}
    (run_dir / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reports), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {args.model} — {args.partition}", "", "Gold-text, zero-shot, Q4_K_M. Chưa ASR/fine-tune. Điểm local chưa được BTC chứng nhận.",
             "Smoke thuộc TRAIN_FIT, không dùng để kết luận model nào tốt hơn trên validation." if args.partition == "smoke" else "Chạy trên VAL_MAIN cố định; kiểm tra sample count để biết full run hay subset.",
             "", f"Số mẫu: {len(reports)} / {manifest['partition_total']}; score local: {summary['score_spec_local']:.2f}/100.",
             "", "| ID | Trạng thái | Calls đúng/gold | Hash đúng | Điểm local | Thời gian text |", "|---|---|---|---|---|---|"]
    for r in reports:
        lines.append(f"| {r['id']} | {r['status']} | {r['matched_calls']}/{r['gold_calls']} | {r['state_match']} | {r['score_spec_local']:.2f} | {r['elapsed_s']:.2f}s |")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("METRICS", json.dumps(summary, ensure_ascii=False), flush=True)
    print("REPORT", run_dir / "SUMMARY.md", flush=True)


if __name__ == "__main__":
    main()
