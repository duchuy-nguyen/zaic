"""Paired held-out analysis of four frozen experiments; report evidence, do not select a winner."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys

from .data import MANIFEST, read_rows, taxonomy, catalog
from .references import PATH as REFERENCE_PATH
from .scoring import score


def load(path):
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    records = [json.loads(s) for s in (path / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    if manifest["status"] != "completed" or [r["id"] for r in records] != manifest["ids"]:
        raise ValueError("Run is incomplete or sample IDs differ: " + str(path))
    if len(records) != len(set(r["id"] for r in records)):
        raise ValueError("Duplicate scored ID")
    # A local 8-call guard is a normal stop with valid executed calls/hash, not a
    # competition timeout or execution failure. Apply BTC's formula to its output.
    # Preserve original run metrics and audit any correction instead of overwriting artifacts.
    original_metrics = dict(metrics)
    corrections = []
    limited = [r for r in records if r["status"] == "call_budget_exhausted" and r["elapsed_s"] <= 30]
    if limited:
        references = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["references"]
        predictions = {r["id"]: r for r in [json.loads(s) for s in (path / "submission.jsonl").read_text(encoding="utf-8").splitlines()]}
        tools = catalog()
        for record in limited:
            updated = score(predictions[record["id"]], references[record["id"]], tools, failed=False)
            corrections.append({"id": record["id"], "original": record["score_spec_local"], "formula_score": updated["score_spec_local"]})
            record.update(updated)
        metrics["score_spec_local"] = statistics.mean(r["score_spec_local"] for r in records)
        metrics["failed_samples_counted_zero"] = sum(r["failure_override"] for r in records)
    return {"path": str(path), "manifest": manifest, "metrics": metrics, "original_metrics": original_metrics,
            "budget_stop_corrections": corrections, "scores": {r["id"]: r for r in records}}


def paired(left, right, groups, *, repeats=10000):
    ids = left["manifest"]["ids"]
    blocks = defaultdict(list)
    wins = losses = ties = 0
    for sample_id in ids:
        delta = right["scores"][sample_id]["score_spec_local"] - left["scores"][sample_id]["score_spec_local"]
        blocks[groups[sample_id]].append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    aggregates = [(sum(values), len(values)) for values in blocks.values()]
    rng = random.Random(20261103)
    samples = []
    for _ in range(repeats):
        picked = rng.choices(aggregates, k=len(aggregates))
        samples.append(sum(v for v, _ in picked) / sum(n for _, n in picked))
    samples.sort()
    return {"delta_right_minus_left": statistics.mean(v for block in blocks.values() for v in block),
            "ci95_group_bootstrap": [samples[int(.025*repeats)], samples[int(.975*repeats)]],
            "right_wins": wins, "right_losses": losses, "ties": ties,
            "samples": len(ids), "groups": len(blocks), "bootstrap_repeats": repeats}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["qwen-gold", "spark-gold", "qwen-asr", "spark-asr"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New report directory, never overwrite")
    args = parser.parse_args()
    runs = {key: load(getattr(args, key.replace("-", "_"))) for key in ["qwen-gold", "spark-gold", "qwen-asr", "spark-asr"]}
    shared = ["ids", "split_sha256", "catalog_sha256", "runtime_py_sha256", "reference_sha256", "code_sha256",
              "prompt_sha256", "scorer", "scorer_assumptions", "runtime_tag", "seed", "thinking", "temperature",
              "context", "max_new_tokens", "max_calls", "max_repairs", "deadline_s", "normalization", "fine_tuned"]
    base = runs["qwen-gold"]["manifest"]
    for key, run in runs.items():
        for field in shared:
            if run["manifest"].get(field) != base.get(field):
                raise ValueError(f"Uncontrolled difference {key}: {field}")
    if runs["qwen-asr"]["manifest"]["asr"] != runs["spark-asr"]["manifest"]["asr"]:
        raise ValueError("ASR providers differ")
    split = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if base["ids"] != split["val_main"]:
        raise ValueError("Expected the complete frozen validation partition")
    comparisons = {
        "text_spark_minus_qwen": paired(runs["qwen-gold"], runs["spark-gold"], split["group_ids"]),
        "audio_spark_minus_qwen": paired(runs["qwen-asr"], runs["spark-asr"], split["group_ids"]),
        "qwen_asr_minus_gold": paired(runs["qwen-gold"], runs["qwen-asr"], split["group_ids"]),
        "spark_asr_minus_gold": paired(runs["spark-gold"], runs["spark-asr"], split["group_ids"]),
    }
    equal_transcripts, mismatched_transcripts, missing_asr = [], [], []
    for sample_id in base["ids"]:
        a = runs["qwen-asr"]["scores"][sample_id]["asr_transcript_sha256"]
        b = runs["spark-asr"]["scores"][sample_id]["asr_transcript_sha256"]
        (missing_asr if a is None or b is None else equal_transcripts if a == b else mismatched_transcripts).append(sample_id)
    rows = {r["id"]: r for r in read_rows()}
    subsets = {
        "zero-call": lambda r: taxonomy(r)["calls"] == 0,
        "ask": lambda r: taxonomy(r)["ask"],
        "history": lambda r: taxonomy(r)["history_turns"] > 0,
        "long-history>=10-turns": lambda r: taxonomy(r)["history_turns"] >= 10,
        "single-call": lambda r: taxonomy(r)["calls"] == 1,
        "multi-call>=3": lambda r: taxonomy(r)["calls"] >= 3,
    }
    slices = {}
    for name, predicate in subsets.items():
        ids = [i for i in base["ids"] if predicate(rows[i])]
        slices[name] = {"n": len(ids), "scores": {key: statistics.mean(run["scores"][i]["score_spec_local"] for i in ids) if ids else None
                                                  for key, run in runs.items()}}
    analysis = {"run_paths": {k: r["path"] for k, r in runs.items()},
                "metrics": {k: r["metrics"] for k, r in runs.items()}, "comparisons": comparisons,
                "original_run_metrics": {k: r["original_metrics"] for k, r in runs.items()},
                "budget_stop_corrections": {k: r["budget_stop_corrections"] for k, r in runs.items()},
                "asr_reproducibility": {"equal": len(equal_transcripts), "different": mismatched_transcripts, "missing": missing_asr},
                "slices": slices, "model_selection": "Not selected; user decides after reviewing these results"}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# So sánh Qwen và Spark trước fine-tune — text chuẩn và audio", "",
             "Toàn bộ 309 mẫu VAL_MAIN giữ riêng. Mỗi model chạy một lượt text chuẩn và một lượt live PhoWhisper-small, không fine-tune/few-shot.",
             "**Metric chính: điểm local bám công thức BTC, thang 0–100, gồm state-hash gate và tool-call F1. Không phải điểm public leaderboard.**",
             "Mọi mẫu lỗi/timeout được giữ trong denominator và chấm 0. Scorer assumptions nằm trong manifest, chưa được BTC chứng nhận.",
             "", "| Model | Đầu vào | N | Điểm chính /100 | Tool F1 | State match | p50 tổng | p95 tổng | Max tổng | Lỗi/timeout |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, run in runs.items():
        m = run["metrics"]
        lines.append(f"| {run['manifest']['model']} | {run['manifest']['mode']} | {m['samples']} | {m['score_spec_local']:.4f} | {m['tool_f1']:.4f} | {m['state_accuracy']:.4f} | {m['latency']['p50_s']:.2f}s | {m['latency']['p95_s']:.2f}s | {m['latency']['max_s']:.2f}s | {m['failed_samples_counted_zero']} |")
    lines += ["", "## So sánh theo từng mẫu và khoảng bất định", "",
              "Chênh lệch tính bằng điểm phần trăm, cùng ID. CI95% dùng 10.000 bootstrap theo conversation/group, không coi timestep là mẫu độc lập.",
              "", "| Phép so | Delta | CI95% | Phía sau hơn/kém/bằng |", "|---|---|---|---|"]
    for key, result in comparisons.items():
        lo, hi = result["ci95_group_bootstrap"]
        lines.append(f"| {key} | {result['delta_right_minus_left']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {result['right_wins']}/{result['right_losses']}/{result['ties']} |")
    lines += ["", "## Kiểm tra ASR giữa hai lượt audio", "",
              f"Transcript hashes giống nhau: {len(equal_transcripts)}/309. Khác: {len(mismatched_transcripts)}. Thiếu do ASR lỗi: {len(missing_asr)}.",
              "ASR chạy live lại ở mỗi mẫu và mỗi model, gồm audio hiện tại và mọi user audio trong history. Không dùng gold fallback hoặc cache để làm đẹp thời gian.",
              "Nếu transcript khác, chênh lệch audio cần được hiểu là kết quả pipeline thực tế; không gán toàn bộ cho riêng LLM.",
              "", "## Theo loại tác vụ", "", "| Slice | N | Qwen text | Spark text | Qwen audio | Spark audio |", "|---|---|---|---|---|---|"]
    for name, value in slices.items():
        scores = " | ".join(f"{value['scores'][key]:.4f}" if value["scores"][key] is not None else "—" for key in runs)
        lines.append(f"| {name} | {value['n']} | {scores} |")
    lines += ["", "## Giới hạn và quyết định", "",
              "- Đây là baseline GGUF Q4_K_M trên RTX 4060 Laptop; thời gian audio đo live gồm ASR/LLM/tool. Chưa chứng nhận A30 hay điểm BTC.",
              "- Startup model/ASR được ghi riêng; cách BTC tính startup cần theo hướng dẫn đóng gói cuối.",
              "- Cùng fixed split/prompt/scorer/runtime/limits, nhưng hai model có tokenizer/native template riêng.",
              "- Fuzzy tokenizer, exact argument-key policy và corner cases là assumptions có version; công thức tổng hợp theo công bố BTC.",
              "- Dừng do guard nội bộ 8 calls trong 30 giây vẫn chấm theo calls/hash thực tế, không tự phạt 0 như timeout. Nếu raw run áp failure override cho guard này, bảng chính đã rescore đúng công thức; chi tiết lưu budget_stop_corrections, artifacts gốc không bị sửa.",
              "- Split có group duplicate/near-text/current-audio; chưa chứng minh speaker-disjoint hoặc hết mọi near-duplicate history.",
              "- Không đổi prompt theo từng mẫu validation trong bốn lượt. VAL_MAIN nay được dùng để chọn model; không gọi là test cuối chưa từng được xem.",
              "- **Chưa chọn model fine-tune.** Người dùng xem điểm audio chính, CI, lỗi và runtime rồi quyết định; ưu thế zero-shot không bảo đảm ưu thế sau SFT.",
              "", "## Artifacts từng lượt", ""]
    for key, run in runs.items():
        summary_link = (Path(run["path"]) / "SUMMARY.md").as_posix()
        lines.append(f"- {key}: [SUMMARY.md]({summary_link}); cùng thư mục có metrics.json, scores.jsonl, traces.jsonl, submission.jsonl và code snapshot.")
    from .audit_experiments import audit
    verification = audit([Path(run["path"]) for run in runs.values()])
    (args.output / "audit.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    lines += ["", "## Kiểm tra sau chạy", "",
              f"Replay hash, calls/traces, token accounting, thinking-off và ASR coverage: {'PASS' if verification['all_checks_passed'] else 'CÓ MISMATCH — cần điều tra trước khi quyết định'}.",
              "Đây là job đánh giá train-only tách biệt sau inference, không phải reset hay probe môi trường trong agent. Chi tiết ở audit.json."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ANALYSIS_REPORT", args.output / "REPORT.md")
    print(json.dumps({"comparisons": comparisons, "asr_equal": len(equal_transcripts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
