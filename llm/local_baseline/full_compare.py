"""Complete 1545-example audio comparison with reusable original-309 artifacts."""
import argparse
from collections import Counter
import csv
from itertools import combinations
import json
from pathlib import Path
import statistics
import sys

from .analyze_experiments import load, paired
from .audit_experiments import audit
from .data import prepare, read_rows, taxonomy
from .expanded_compare import OLD_GEMMA, merged, summary
from .extra_metrics import diagnostics
from .full_selection import PATH as SELECTION, create, expected_full_source
from .qwen35_compare import check_sources as check_original_sources
from .setup_local import ROOT, sha256

OLD_4B = ROOT / "runs/20261104T143403Z_full_qwen35_asr_688fa7"
LABELS = {"qwen35": "Qwen3.5-4B", "qwen35_2b": "Qwen3.5-2B", "gemma": "Gemma 4 E2B"}
FIELDS = ["split_sha256", "catalog_sha256", "runtime_py_sha256", "reference_sha256", "prompt_sha256", "scorer",
          "scorer_assumptions", "runtime_tag", "seed", "thinking", "temperature", "context", "max_new_tokens",
          "max_calls", "max_repairs", "deadline_s", "normalization", "fine_tuned", "asr", "mode"]


def check_full_sources(directory):
    for filename in ("agent.py", "backend.py", "data.py", "scoring.py", "speech.py", "experiment.py"):
        before = (OLD_4B / "code_snapshot" / filename).read_text(encoding="utf-8")
        after = (directory / "code_snapshot" / filename).read_text(encoding="utf-8")
        if filename == "backend.py":
            after = after.replace('    "qwen35_2b": "Qwen3.5-2B-Q4_K_M.gguf",\n', '')
        elif filename == "experiment.py":
            before = expected_full_source(before)
        if before != after:
            raise ValueError("Unexpected inference/policy change: " + filename)


def combine_diagnostics(paths):
    parts = [diagnostics(path) for path in paths]
    n = sum(p["samples"] for p in parts)
    errors = {key: sum(p["asr_error_counts"][key] for p in parts) for key in parts[0]["asr_error_counts"]}
    nonempty = sum(p["nonempty_gold_samples"] for p in parts)
    return {"samples": n, "exact_tool_sequence_accuracy": sum(p["exact_tool_sequence_accuracy"]*p["samples"] for p in parts)/n,
            "nonempty_gold_state_accuracy": sum((p["nonempty_gold_state_accuracy"] or 0)*p["nonempty_gold_samples"] for p in parts)/nonempty,
            "nonempty_gold_samples": nonempty,
            "current_asr_wer": errors["word_edits"]/errors["reference_words"] if errors["reference_words"] else None,
            "current_asr_cer": errors["character_edits"]/errors["reference_characters"] if errors["reference_characters"] else None,
            "asr_error_counts": errors,
            "missing_current_transcripts": [i for p in parts for i in p["missing_current_transcripts"]],
            "reasoning_samples": sorted({i for p in parts for i in p["reasoning_samples"]})}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    for model in LABELS:
        parser.add_argument("--" + model, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = create()
    split = prepare()
    previous = {"qwen35": load(OLD_4B), "gemma": load(OLD_GEMMA)}
    fresh = {model: load(getattr(args, model)) for model in LABELS}
    check_original_sources(OLD_GEMMA, OLD_4B)
    base = previous["qwen35"]["manifest"]
    for model, run in fresh.items():
        expected = selection["all_train"] if model == "qwen35_2b" else selection["remaining_1236"]
        if run["manifest"]["ids"] != expected or run["manifest"]["full_selection_manifest_sha256"] != sha256(SELECTION):
            raise ValueError("Incomplete or different full selection: " + model)
        for field in FIELDS:
            if run["manifest"].get(field) != base.get(field):
                raise ValueError(f"Changed configuration {model}: {field}")
        check_full_sources(Path(run["path"]))
        if run["manifest"]["code_sha256"] != fresh["qwen35"]["manifest"]["code_sha256"]:
            raise ValueError("Source drift between full jobs")
        if model in previous and run["manifest"]["weight"] != previous[model]["manifest"]["weight"]:
            raise ValueError("Weights changed: " + model)
    for run in previous.values():
        for field in FIELDS:
            if run["manifest"].get(field) != base.get(field):
                raise ValueError("Original 309 configuration differs: " + field)
    full = {model: merged(previous[model], run) if model in previous else run for model, run in fresh.items()}
    for model, run in full.items():
        if sorted(run["scores"]) != selection["all_train"]:
            raise ValueError("Full set must contain each of 1545 IDs once: " + model)
    cohorts = {"full_1545": selection["all_train"], "original_309": selection["original_309"],
               "remaining_1236": selection["remaining_1236"],
               "remaining_without_smoke_1224": sorted(set(selection["remaining_1236"])-set(selection["smoke_included"]))}
    cohort_runs = {name: {model: {"manifest": {"ids": ids}, "scores": {i: run["scores"][i] for i in ids}} for model, run in full.items()} for name, ids in cohorts.items()}
    metrics = {name: {model: summary(run) for model, run in runs.items()} for name, runs in cohort_runs.items()}
    comparisons = {name: {f"{right}_minus_{left}": paired(runs[left], runs[right], split["group_ids"])
                          for left, right in combinations(LABELS, 2)} for name, runs in cohort_runs.items() if name != "remaining_without_smoke_1224"}
    transcripts = {}
    for model in ("qwen35_2b", "gemma"):
        different, missing = [], []
        for sample_id in selection["all_train"]:
            a = full["qwen35"]["scores"][sample_id]["asr_transcript_sha256"]
            b = full[model]["scores"][sample_id]["asr_transcript_sha256"]
            if a is None or b is None:
                missing.append(sample_id)
            elif a != b:
                different.append(sample_id)
        transcripts[model] = {"equal": 1545-len(different)-len(missing), "different": different, "missing": missing}
    rows = {r["id"]: r for r in read_rows()}
    predicates = {"zero-call": lambda r: taxonomy(r)["calls"] == 0,
                  "ask": lambda r: taxonomy(r)["ask"], "single-call": lambda r: taxonomy(r)["calls"] == 1,
                  "multi-call>=3": lambda r: taxonomy(r)["calls"] >= 3,
                  "history": lambda r: taxonomy(r)["history_turns"] > 0,
                  "long-history>=10": lambda r: taxonomy(r)["history_turns"] >= 10}
    slices = {}
    for name, predicate in predicates.items():
        ids = [i for i in selection["all_train"] if predicate(rows[i])]
        slices[name] = {"n": len(ids), "scores": {model: statistics.mean(run["scores"][i]["score_spec_local"] for i in ids) if ids else None for model, run in full.items()}}
    paths = {model: ([Path(previous[model]["path"])] if model in previous else [])+[Path(run["path"])] for model, run in fresh.items()}
    extra = {model: combine_diagnostics(value) for model, value in paths.items()}
    verification = audit([Path(run["path"]) for run in fresh.values()])
    args.output.mkdir(parents=True, exist_ok=False)
    analysis = {"metrics": metrics, "comparisons": comparisons, "extra_metrics": extra, "asr_transcripts_vs_4b": transcripts,
                "slices": slices, "audit": verification, "not_official_score": True,
                "run_paths": {model: [str(p) for p in value] for model, value in paths.items()},
                "selection_sha256": sha256(SELECTION), "labels": LABELS,
                "budget_stop_corrections": {model: run["budget_stop_corrections"] for model, run in fresh.items()}}
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["model", "samples", "score_100", "tool_f1_percent", "strict_tool_f1_percent", "state_match_percent",
               "exact_sequence_percent", "nonempty_gold_state_percent", "wer_current_percent", "cer_current_percent",
               "p50_total_s", "p95_total_s", "max_total_s", "p50_asr_s", "p95_asr_s", "p50_agent_s", "p95_agent_s",
               "timeouts", "failures", "call_budget_stops", "reasoning_samples"]
    with (args.output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for model, m in metrics["full_1545"].items():
            e = extra[model]; scores = list(full[model]["scores"].values())
            writer.writerow(dict(zip(columns, [LABELS[model], 1545, m["score"], 100*m["tool_f1"],
                100*statistics.mean(r["strict_tool_f1"] for r in scores), 100*m["state_accuracy"],
                100*e["exact_tool_sequence_accuracy"], 100*e["nonempty_gold_state_accuracy"],
                100*e["current_asr_wer"] if e["current_asr_wer"] is not None else '',
                100*e["current_asr_cer"] if e["current_asr_cer"] is not None else '',
                m["latency"]["p50_s"], m["latency"]["p95_s"], m["latency"]["max_s"],
                m["asr_latency"]["p50_s"], m["asr_latency"]["p95_s"], m["agent_latency"]["p50_s"], m["agent_latency"]["p95_s"],
                m["statuses"].get("timeout",0), m["failures_counted_zero"], m["statuses"].get("call_budget_exhausted",0), len(e["reasoning_samples"])])))
    lines = ["# Full zero-shot: Qwen3.5-4B, Qwen3.5-2B và Gemma 4 E2B", "",
             "Đủ 1.545 mẫu train có nhãn cho mỗi model, pipeline audio → PhoWhisper-small live (cả user history) → LLM/tool. Giữ nguyên weights Q4_K_M, prompt, seed/greedy, context 16.384, max 512 tokens, 8 calls/2 repairs, 30s/sample. Không fine-tune/RAG/few-shot.",
             "309 mẫu đã có của 4B/Gemma được giữ nguyên; bổ sung 1.236 mẫu/model. 2B chạy mới 1.545 mẫu. Tất cả ID xuất hiện đúng một lần/model; lỗi không bị bỏ khỏi denominator.",
             "**Điểm local-spec-v1 bám công thức BTC, không phải điểm leaderboard hoặc scorer được BTC chứng nhận.**", "",
             "| Model | N | Điểm /100 | Tool F1 | State đúng | p50 tổng | p95 tổng | Timeout | Lỗi chấm 0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model, m in metrics["full_1545"].items():
        lines.append(f"| {LABELS[model]} | 1545 | {m['score']:.4f} | {100*m['tool_f1']:.2f}% | {100*m['state_accuracy']:.2f}% | {m['latency']['p50_s']:.2f}s | {m['latency']['p95_s']:.2f}s | {m['statuses'].get('timeout',0)} | {m['failures_counted_zero']} |")
    lines += ["", "## Giải thích metrics", "",
              "- **Điểm chính /100**: 100 × trung bình điểm mẫu. Gold có lệnh: hash sai hoặc không match call nào → 0; hash đúng → tool-call F1. Gold rỗng: pred rỗng → 1, còn lại 0. Timeout/lỗi xử lý → 0. Không lấy trung bình các metric phụ để thay công thức này.",
              "- **Tool F1**: mức khớp calls/arguments, one-to-one có giữ duplicate; không xét thứ tự matching. Tên/args exact trừ fuzzy theo schema với token-F1 >0,6. Thứ tự thực thi vẫn ảnh hưởng state hash.",
              "- **State đúng**: tỷ lệ full state hash trùng reference replay. State đúng chưa đủ chứng minh hoàn thành yêu cầu (ví dụ lookup không đổi state); vì vậy phải xem điểm chính và F1 cùng nhau.",
              "- **Strict F1 / exact sequence**: strict F1 không fuzzy; exact sequence yêu cầu đúng thứ tự và toàn bộ calls/args. Đây là chẩn đoán bổ sung, không phải điểm BTC.",
              "- **p50 / p95**: trung vị và mốc 95% mẫu hoàn thành không lâu hơn thời gian đó. Tổng gồm live ASR, chuẩn bị env/input, LLM/tool; startup ghi riêng. Lỗi/timeout vẫn có trong phân phối thời gian.",
              "- **WER/CER**: tỷ lệ edit lỗi phiên âm audio lượt hiện tại, micro corpus. NFC/lowercase/bỏ punctuation/giữ dấu; WER theo token cách trắng tiếng Việt, CER bỏ khoảng trắng. Không chuẩn hóa tương đương số/giờ/ngữ nghĩa. Là chẩn đoán ASR chung, không dùng riêng để xếp hạng LLM. Missing transcript được ghi rõ.",
              "- **Lỗi chấm 0** gồm timeout và lỗi xử lý/thực thi/định dạng. Guard nội bộ 8 calls khi còn trong hạn vẫn chấm calls/hash thực tế; raw artifacts giữ nguyên và rescore được audit.",
              "- **CI95%** dùng 10.000 bootstrap theo duplicate/near-duplicate group. Khoảng chứa 0 nghĩa là chưa phân định chắc chênh lệch trên phép đo này; không bảo đảm kết quả sau fine-tune hoặc hidden test.",
              "", "## Chỉ số bổ sung", "", "| Model | Strict F1 | Exact sequence | WER current | CER current | Reasoning samples |", "|---|---|---|---|---|---|"]
    for model, e in extra.items():
        strict = statistics.mean(r["strict_tool_f1"] for r in full[model]["scores"].values())
        wer = f"{100*e['current_asr_wer']:.2f}%" if e['current_asr_wer'] is not None else 'missing'
        cer = f"{100*e['current_asr_cer']:.2f}%" if e['current_asr_cer'] is not None else 'missing'
        lines.append(f"| {LABELS[model]} | {100*strict:.2f}% | {100*e['exact_tool_sequence_accuracy']:.2f}% | {wer} | {cer} | {len(e['reasoning_samples'])} |")
    lines += ["", "## So sánh theo cùng mẫu", "", "| Bộ mẫu | Phép so | Delta | CI95% | Wins/losses/ties |", "|---|---|---|---|---|"]
    for cohort, values in comparisons.items():
        for name, value in values.items():
            lo, hi = value["ci95_group_bootstrap"]
            lines.append(f"| {cohort} | {name} | {value['delta_right_minus_left']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {value['right_wins']}/{value['right_losses']}/{value['ties']} |")
    lines += ["", "## Theo loại tác vụ trên toàn bộ 1.545 mẫu", "", "| Slice | N | 4B | 2B | Gemma |", "|---|---|---|---|---|"]
    for name, value in slices.items():
        scores = ' | '.join(f"{value['scores'][model]:.4f}" if value['scores'][model] is not None else '—' for model in LABELS)
        lines.append(f"| {name} | {value['n']} | {scores} |")
    lines += ["", "## Các phần dữ liệu", "", "| Cohort | N | 4B | 2B | Gemma |", "|---|---|---|---|---|"]
    for name, values in metrics.items():
        lines.append(f"| {name} | {values['qwen35']['n']} | {values['qwen35']['score']:.4f} | {values['qwen35_2b']['score']:.4f} | {values['gemma']['score']:.4f} |")
    lines += ["", "## Kiểm tra và giới hạn", "",
              "- Đối chiếu ASR hashes với 4B: " + json.dumps(transcripts, ensure_ascii=False),
              "- Audit các lượt bổ sung: " + ("PASS" if verification['all_checks_passed'] else "Có cờ cần đọc trong analysis.json; không bỏ mẫu hoặc rerun chọn lọc."),
              "- Thinking được yêu cầu tắt nhưng vẫn kiểm output thực tế. Mọi reasoning phát sinh đều được ghi, token/time thực tế vẫn tính.",
              "- Fingerprint/snapshot kiểm tra prompt, ASR, agent, scorer, data/runtime và weights; runner chỉ thêm chọn toàn tập và metadata, backend chỉ thêm tên model 2B.",
              "- Đây là đánh giá zero-shot trên train gốc, gồm cả 12 smoke đã dùng phát triển prompt; không gọi là blind test cuối. Tách cohort không smoke để minh bạch. Các mẫu vẫn có thể dùng fine-tune sau này.",
              "- Scorer còn assumptions về fuzzy tokenizer/exact keys; không phải official executable scorer. Grouping là heuristic, không speaker-disjoint.",
              "- Cùng RTX4060, khác thời điểm/tải nền/nhiệt/xung/chế độ điện. ASR latency cũng có thể khác; không quy toàn bộ chênh lệch end-to-end cho LLM. Chưa chứng nhận runtime A30 hoặc cách BTC tính startup.",
              "- Chỉ local, không tự train/push/upload. Ưu thế zero-shot chưa bảo đảm ưu thế sau fine-tune.",
              "", "Bảng đầy đủ: [metrics.csv](metrics.csv). Chi tiết audit/cấu hình/CI: [analysis.json](analysis.json)."]
    (args.output / "REPORT.md").write_text('\n'.join(lines)+'\n', encoding="utf-8")
    print("FULL_REPORT", args.output / "REPORT.md", flush=True)
    print(json.dumps({"full_metrics": metrics['full_1545'], "comparisons": comparisons['full_1545'], "asr": transcripts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
