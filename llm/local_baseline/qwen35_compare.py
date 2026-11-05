"""Compare Qwen3.5 against preserved Qwen3/Gemma audio validation runs."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

from .analyze_experiments import load, paired
from .audit_experiments import audit
from .compare_three import verify_sources
from .data import prepare, read_rows, taxonomy
from .expanded_compare import OLD_QWEN, OLD_GEMMA, expected_expanded_source, summary
from .extra_metrics import diagnostics


def check_sources(old_path, new_path):
    old = Path(old_path) / "code_snapshot"
    new = Path(new_path) / "code_snapshot"
    for filename in ("agent.py", "backend.py", "data.py", "scoring.py", "speech.py", "experiment.py"):
        a = (old / filename).read_text(encoding="utf-8")
        b = (new / filename).read_text(encoding="utf-8")
        if filename == "experiment.py":
            a = expected_expanded_source(a)
        elif filename == "backend.py":
            b = b.replace('    "qwen35": "Qwen3.5-4B-Q4_K_M.gguf",\n', '')
        if a != b:
            raise ValueError("Unexpected code change: " + filename)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {"qwen3": load(OLD_QWEN), "gemma": load(OLD_GEMMA), "qwen35": load(args.qwen35)}
    split = prepare()
    shared = ["ids", "split_sha256", "catalog_sha256", "runtime_py_sha256", "reference_sha256", "prompt_sha256", "scorer",
              "scorer_assumptions", "runtime_tag", "seed", "thinking", "temperature", "context", "max_new_tokens",
              "max_calls", "max_repairs", "deadline_s", "normalization", "fine_tuned", "asr", "mode", "partition"]
    for name, run in runs.items():
        if run["manifest"]["ids"] != split["val_main"]:
            raise ValueError("Not complete original 309-sample validation: " + name)
        for field in shared:
            if run["manifest"].get(field) != runs["qwen3"]["manifest"].get(field):
                raise ValueError(f"Configuration differs {name}: {field}")
    verify_sources(runs["qwen3"], runs["gemma"])
    check_sources(OLD_GEMMA, args.qwen35)
    metrics = {model: summary(run) for model, run in runs.items()}
    comparisons = {f"qwen35_minus_{model}": paired(runs[model], runs["qwen35"], split["group_ids"]) for model in ("qwen3", "gemma")}
    transcripts = {}
    for model in ("gemma", "qwen35"):
        different, missing = [], []
        for sample_id in split["val_main"]:
            a = runs["qwen3"]["scores"][sample_id]["asr_transcript_sha256"]
            b = runs[model]["scores"][sample_id]["asr_transcript_sha256"]
            if a is None or b is None:
                missing.append(sample_id)
            elif a != b:
                different.append(sample_id)
        transcripts[model] = {"equal": 309-len(different)-len(missing), "different": different, "missing": missing}
    rows = {r["id"]: r for r in read_rows()}
    predicates = {"zero-call": lambda r: taxonomy(r)["calls"] == 0,
                  "ask": lambda r: taxonomy(r)["ask"], "single-call": lambda r: taxonomy(r)["calls"] == 1,
                  "multi-call>=3": lambda r: taxonomy(r)["calls"] >= 3,
                  "history": lambda r: taxonomy(r)["history_turns"] > 0,
                  "long-history>=10": lambda r: taxonomy(r)["history_turns"] >= 10}
    slices = {}
    for name, predicate in predicates.items():
        ids = [i for i in split["val_main"] if predicate(rows[i])]
        slices[name] = {"n": len(ids), "scores": {model: statistics.mean(run["scores"][i]["score_spec_local"] for i in ids) if ids else None for model, run in runs.items()}}
    verification = audit([args.qwen35])
    extra = {model: diagnostics(Path(run["path"])) for model, run in runs.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    analysis = {"metrics": metrics, "comparisons": comparisons, "asr_transcripts_vs_qwen3": transcripts,
                "slices": slices, "audit": verification, "extra_metrics": extra, "not_official_score": True,
                "run_paths": {name: run["path"] for name, run in runs.items()},
                "budget_stop_corrections": {name: run["budget_stop_corrections"] for name, run in runs.items()},
                "fine_tuning": False, "expanded_600": "Remains paused; not part of this comparison"}
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["model", "samples", "score_spec_local_100", "tool_f1_percent", "strict_tool_f1_percent", "state_accuracy_percent",
                  "exact_tool_sequence_percent", "current_asr_wer_percent", "current_asr_cer_percent", "total_p50_s", "total_p95_s",
                  "failures_counted_zero", "timeout_samples", "reasoning_samples"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, m in metrics.items():
            run = runs[model]; e = extra[model]
            writer.writerow({"model": model, "samples": m["n"], "score_spec_local_100": m["score"],
                             "tool_f1_percent": 100*m["tool_f1"], "strict_tool_f1_percent": 100*run["metrics"]["strict_tool_f1"],
                             "state_accuracy_percent": 100*m["state_accuracy"], "exact_tool_sequence_percent": 100*e["exact_tool_sequence_accuracy"],
                             "current_asr_wer_percent": 100*e["current_asr_wer"] if e["current_asr_wer"] is not None else '',
                             "current_asr_cer_percent": 100*e["current_asr_cer"] if e["current_asr_cer"] is not None else '',
                             "total_p50_s": m["latency"]["p50_s"], "total_p95_s": m["latency"]["p95_s"],
                             "failures_counted_zero": m["failures_counted_zero"], "timeout_samples": m["statuses"].get("timeout", 0),
                             "reasoning_samples": len(e["reasoning_samples"])})
    lines = ["# Zero-shot audio: Qwen3.5-4B so với Qwen3 và Gemma 4 E2B", "",
             "Cùng 309 mẫu VAL_MAIN, cùng PhoWhisper-small live cả current/history, cùng agent/prompt/tool/scorer và giới hạn. Không fine-tune/few-shot/RAG. Điểm local bám công thức BTC, không phải leaderboard.", "",
             "Điểm chính = 100 × trung bình điểm từng mẫu. Gold có lệnh: state hash sai hoặc không match được call nào thì điểm 0; nếu hash đúng thì lấy tool-call F1. Gold không có lệnh: dự đoán không có lệnh được 1, có lệnh được 0. Timeout/lỗi xử lý được chấm 0. F1 và state match là chẩn đoán riêng, không lấy trung bình tùy ý để thay công thức chính.",
             "Tool matching giữ duplicate, không xét thứ tự để tính F1; tên/args exact trừ tham số fuzzy theo schema với token-F1 >0,6. Thứ tự thực thi vẫn quyết định trạng thái cuối. Chi tiết assumptions tokenizer/exact-key nằm trong manifest local-spec-v1.", "",
             "| Model Q4_K_M | Điểm /100 | Tool F1 (0–1) | State match (0–1) | p50 tổng | p95 tổng | Lỗi chấm 0 |",
             "|---|---|---|---|---|---|---|"]
    for model, m in metrics.items():
        lines.append(f"| {model} | {m['score']:.4f} | {m['tool_f1']:.4f} | {m['state_accuracy']:.4f} | {m['latency']['p50_s']:.2f}s | {m['latency']['p95_s']:.2f}s | {m['failures_counted_zero']} |")
    lines += ["", "## Chênh lệch theo cùng mẫu", "", "| Phép so | Delta điểm | CI95% bootstrap group | Wins/losses/ties |", "|---|---|---|---|"]
    for name, value in comparisons.items():
        low, high = value["ci95_group_bootstrap"]
        lines.append(f"| {name} | {value['delta_right_minus_left']:+.4f} | [{low:+.4f}, {high:+.4f}] | {value['right_wins']}/{value['right_losses']}/{value['ties']} |")
    lines += ["", "## Theo tác vụ", "", "| Slice | N | Qwen3 | Gemma | Qwen3.5 |", "|---|---|---|---|---|"]
    for name, value in slices.items():
        scores = ' | '.join(f"{value['scores'][model]:.4f}" if value['scores'][model] is not None else '—' for model in runs)
        lines.append(f"| {name} | {value['n']} | {scores} |")
    lines += ["", "## Chỉ số bổ sung", "", "| Model | Chuỗi tool exact | Strict tool F1 | ASR WER current | ASR CER current |", "|---|---|---|---|---|"]
    for model, e in extra.items():
        wer = f"{100*e['current_asr_wer']:.2f}%" if e['current_asr_wer'] is not None else 'missing'
        cer = f"{100*e['current_asr_cer']:.2f}%" if e['current_asr_cer'] is not None else 'missing'
        lines.append(f"| {model} | {100*e['exact_tool_sequence_accuracy']:.2f}% | {100*runs[model]['metrics']['strict_tool_f1']:.2f}% | {wer} | {cer} |")
    lines += ["", "WER/CER chỉ đo audio lượt hiện tại, corpus micro edit rate; NFC/lowercase, bỏ punctuation, giữ dấu; CER bỏ khoảng trắng. WER theo đơn vị cách trắng tiếng Việt (thường gần âm tiết), không phải chuẩn chấm BTC. Exact sequence xét đúng thứ tự/tất cả tham số; F1 chính theo spec không xét thứ tự matching. File metrics.csv chứa bảng đầy đủ.",
              "", "## Tách thời gian ASR và agent", "", "| Model | ASR p50/p95 | Agent p50/p95 | Trạng thái |", "|---|---|---|---|"]
    for model, m in metrics.items():
        lines.append(f"| {model} | {m['asr_latency']['p50_s']:.2f}/{m['asr_latency']['p95_s']:.2f}s | {m['agent_latency']['p50_s']:.2f}/{m['agent_latency']['p95_s']:.2f}s | {m['statuses']} |")
    lines += ["", "## Audit và giới hạn", "",
              "- ASR hashes so với Qwen3: " + json.dumps(transcripts, ensure_ascii=False),
              "- Audit Qwen3.5 replay/hash, executed calls, tokens, reasoning, ASR coverage: " + ("PASS" if verification["all_checks_passed"] else "Có cờ cần kiểm tra trong analysis.json; không xóa hoặc rerun chọn lọc mẫu."),
              "- Đã kiểm exact source snapshots: thay đổi chỉ registry model và phần lựa chọn ID/metadata đã thêm trước đó; không đổi inference/prompt/scorer/ASR.",
              "- Giữ tất cả 309 mẫu trong denominator; timeout/error chấm 0. Guard 8 calls trong thời hạn chấm calls/hash thực tế, giữ raw artifacts.",
              "- Thinking được yêu cầu tắt nhưng vẫn audit output thực tế. Native templates/tokenizer/kiến trúc riêng cho từng model.",
              "- Cùng RTX4060, khác thời điểm/tải nền/xung/chế độ điện; ASR cũng có thể thay đổi tốc độ. Không quy toàn bộ chênh lệch end-to-end cho LLM. Startup ghi riêng; chưa chứng nhận runtime A30.",
              "- Scorer local-spec-v1 còn assumptions fuzzy/exact-key; không phải BTC scorer. Grouping gần trùng là heuristic, không chứng minh speaker-disjoint.",
              "- VAL_MAIN đã dùng để shortlist model, không gọi đây là blind test cuối. Ưu thế zero-shot chưa bảo đảm ưu thế sau fine-tune.",
              "- Campaign 600 mẫu vẫn tạm dừng ở 28 mẫu Qwen3, không tính vào bảng này. Chưa tự train hoặc chạy tiếp 600 mẫu."]
    (args.output / "REPORT.md").write_text('\n'.join(lines) + '\n', encoding="utf-8")
    print("QWEN35_REPORT", args.output / "REPORT.md", flush=True)
    print(json.dumps({"metrics": metrics, "comparisons": comparisons, "transcripts": transcripts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
