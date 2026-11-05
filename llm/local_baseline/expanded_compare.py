"""Paired analysis of the frozen new 600 plus the previous 309 audio samples."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

from .analyze_experiments import load, paired
from .audit_experiments import audit
from .compare_three import verify_sources
from .data import prepare, read_rows, taxonomy
from .expanded_selection import PATH as SELECTION, validate_selection
from .setup_local import ROOT, sha256

OLD_QWEN = ROOT / "runs/20261104T113905Z_full_qwen_asr_039c76"
OLD_GEMMA = ROOT / "runs/20261104T134837Z_full_gemma_asr_613544"


def expected_expanded_source(source):
    replacements = [
        ('choices=["val-main", "smoke"]', 'choices=["val-main", "smoke", "expanded"]'),
        ('    args = parser.parse_args()\n', '    parser.add_argument("--ids-manifest", type=Path)\n    args = parser.parse_args()\n'),
        ('    if args.limit is not None:\n', '''    if args.partition == "expanded":
        if args.ids_manifest is None or args.limit is not None:
            parser.error("expanded requires --ids-manifest and the complete frozen selection (no --limit)")
        from .expanded_selection import validate_selection
        ids = validate_selection(json.loads(args.ids_manifest.read_text(encoding="utf-8")), split)
    elif args.ids_manifest is not None:
        parser.error("--ids-manifest is only valid with --partition expanded")
    if args.limit is not None:
'''),
        ('    code = {name:', '    if args.partition == "expanded":\n        critical.append("expanded_selection.py")\n    code = {name:'),
        ('    run_dir = args.resume.resolve()', '    if args.partition == "expanded":\n        common["selection_manifest_sha256"] = sha256(args.ids_manifest)\n    run_dir = args.resume.resolve()'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("Unexpected original experiment source")
        source = source.replace(before, after)
    return source


def check_sources(old, new):
    for name in old["manifest"]["code_sha256"]:
        a = (Path(old["path"]) / "code_snapshot" / name).read_text(encoding="utf-8")
        b = (Path(new["path"]) / "code_snapshot" / name).read_text(encoding="utf-8")
        if name == "experiment.py":
            a = expected_expanded_source(a)
        if a != b:
            raise ValueError("Unexpected inference/source change: " + name)


def merged(old, new):
    if set(old["scores"]) & set(new["scores"]):
        raise ValueError("Overlapping old/new sample IDs")
    scores = {**old["scores"], **new["scores"]}
    return {"manifest": {"ids": sorted(scores)}, "scores": scores}


def summary(run):
    rows = list(run["scores"].values())
    def latency(field):
        values = sorted(r[field] for r in rows)
        return {"p50_s": statistics.median(values), "p95_s": values[math.ceil(.95*len(values))-1], "max_s": max(values)}
    return {"n": len(rows), "score": statistics.mean(r["score_spec_local"] for r in rows),
            "tool_f1": statistics.mean(r["tool_f1"] for r in rows),
            "state_accuracy": statistics.mean(r["state_match"] for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "failures_counted_zero": sum(r["failure_override"] for r in rows),
            "latency": latency("elapsed_s"), "asr_latency": latency("asr_s"), "agent_latency": latency("llm_and_agent_s")}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = prepare()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    selected = validate_selection(selection, split)
    old = {"qwen": load(OLD_QWEN), "gemma": load(OLD_GEMMA)}
    new = {"qwen": load(args.qwen), "gemma": load(args.gemma)}
    shared = ["split_sha256", "catalog_sha256", "runtime_py_sha256", "reference_sha256", "prompt_sha256", "scorer",
              "scorer_assumptions", "runtime_tag", "seed", "thinking", "temperature", "context", "max_new_tokens",
              "max_calls", "max_repairs", "deadline_s", "normalization", "fine_tuned", "asr"]
    for model in ("qwen", "gemma"):
        for field in shared:
            if new[model]["manifest"][field] != old[model]["manifest"][field]:
                raise ValueError(f"Changed {model} configuration: {field}")
            if new[model]["manifest"][field] != new["qwen"]["manifest"][field]:
                raise ValueError("Different model configurations: " + field)
        if new[model]["manifest"]["ids"] != selected or new[model]["manifest"]["selection_manifest_sha256"] != sha256(SELECTION):
            raise ValueError("Different/incomplete expansion selection")
        if new[model]["manifest"]["weight"] != old[model]["manifest"]["weight"]:
            raise ValueError("Different weights")
        check_sources(old["gemma"], new[model])
    verify_sources(old["qwen"], old["gemma"])
    if new["qwen"]["manifest"]["code_sha256"] != new["gemma"]["manifest"]["code_sha256"]:
        raise ValueError("Source changes during new paired runs")
    combined = {model: merged(old[model], new[model]) for model in old}
    cohorts = {"previous_309": old, "new_600": new, "combined_909": combined}
    comparisons = {name: paired(value["qwen"], value["gemma"], split["group_ids"]) for name, value in cohorts.items()}
    metrics = {name: {model: summary(run) for model, run in value.items()} for name, value in cohorts.items()}
    asr_checks = {}
    for name, value in cohorts.items():
        different, missing = [], []
        for sample_id in value["qwen"]["manifest"]["ids"]:
            a = value["qwen"]["scores"][sample_id]["asr_transcript_sha256"]
            b = value["gemma"]["scores"][sample_id]["asr_transcript_sha256"]
            if a is None or b is None:
                missing.append(sample_id)
            elif a != b:
                different.append(sample_id)
        asr_checks[name] = {"different": different, "missing": missing,
                            "equal": len(value["qwen"]["scores"]) - len(different) - len(missing)}
    rows = {r["id"]: r for r in read_rows()}
    predicates = {"zero-call": lambda r: taxonomy(r)["calls"] == 0,
                  "ask": lambda r: taxonomy(r)["ask"], "single-call": lambda r: taxonomy(r)["calls"] == 1,
                  "multi-call>=3": lambda r: taxonomy(r)["calls"] >= 3,
                  "history": lambda r: taxonomy(r)["history_turns"] > 0,
                  "long-history>=10": lambda r: taxonomy(r)["history_turns"] >= 10}
    slices = {}
    for name, predicate in predicates.items():
        ids = [i for i in combined["qwen"]["manifest"]["ids"] if predicate(rows[i])]
        slices[name] = {"n": len(ids), "scores": {model: statistics.mean(run["scores"][i]["score_spec_local"] for i in ids) if ids else None for model, run in combined.items()}}
    verification = audit([args.qwen, args.gemma])
    args.output.mkdir(parents=True, exist_ok=False)
    analysis = {"metrics": metrics, "paired_gemma_minus_qwen": comparisons, "asr_checks": asr_checks,
                "slices_combined": slices, "audit": verification, "selection_sha256": sha256(SELECTION),
                "new_run_paths": {k: v["path"] for k, v in new.items()},
                "original_run_paths": {k: v["path"] for k, v in old.items()},
                "budget_stop_corrections": {k: v["budget_stop_corrections"] for k, v in new.items()},
                "not_official_score": True, "no_fine_tuning": True}
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Mở rộng zero-shot Qwen/Gemma: 909 mẫu audio mỗi model", "",
             "600 mẫu mới được chọn và chốt trước inference, không trùng ID/group với 309 mẫu cũ hoặc smoke. Không fine-tune, few-shot, RAG hay chỉnh prompt theo kết quả. Các mẫu vẫn có thể dùng train sau này.",
             "Điểm local-spec-v1 theo công thức BTC; không phải leaderboard. Giữ mọi lỗi/timeout trong denominator.", "",
             "| Bộ mẫu | N/model | Qwen /100 | Gemma /100 | Gemma − Qwen | CI95% theo group |", "|---|---|---|---|---|---|"]
    for name in cohorts:
        m = metrics[name]; c = comparisons[name]; lo, hi = c["ci95_group_bootstrap"]
        lines.append(f"| {name} | {m['qwen']['n']} | {m['qwen']['score']:.4f} | {m['gemma']['score']:.4f} | {c['delta_right_minus_left']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "600 mới là bằng chứng bổ sung tách biệt; 909 là mô tả gộp để ưu tiên model. Đã dùng 309 cũ để chọn hai ứng viên nên không gọi kết quả gộp là test hoàn toàn mù. CI bootstrap là xấp xỉ, không bảo đảm phân định được model có chênh lệch rất nhỏ.",
              "", "## Tốc độ và lỗi trên 600 mẫu mới", "",
              "| Model | p50 tổng | p95 tổng | ASR p50 | Agent p50 | Lỗi chấm 0 | Trạng thái |", "|---|---|---|---|---|---|---|"]
    for model, m in metrics["new_600"].items():
        lines.append(f"| {model} | {m['latency']['p50_s']:.2f}s | {m['latency']['p95_s']:.2f}s | {m['asr_latency']['p50_s']:.2f}s | {m['agent_latency']['p50_s']:.2f}s | {m['failures_counted_zero']} | {m['statuses']} |")
    lines += ["", "## Theo loại tác vụ trên 909 mẫu", "", "| Slice | N | Qwen | Gemma |", "|---|---|---|---|"]
    for name, value in slices.items():
        lines.append(f"| {name} | {value['n']} | {value['scores']['qwen']:.4f} | {value['scores']['gemma']:.4f} |")
    lines += ["", "## Kiểm tra và giới hạn", "",
              "- Input ASR đối chiếu: " + json.dumps(asr_checks, ensure_ascii=False),
              "- Audit replay/hash, calls/traces, token count, reasoning và ASR coverage: " + ("PASS" if verification["all_checks_passed"] else "Có cờ cần đọc trong analysis.json; không tự xóa hoặc rerun chọn lọc mẫu lỗi."),
              "- Tắt thinking được yêu cầu; model vẫn có thể phát sinh reasoning. Mọi trường hợp được giữ, thời gian/token thực tế vẫn tính. Lượt Gemma text cũ có một ngoại lệ reasoning, nhưng text không nằm trong phép so audio mở rộng này.",
              "- Agent/backend/scorer/ASR giữ nguyên; experiment chỉ thêm chọn ID và metadata. Comparator kiểm source snapshots và fingerprint cấu hình. Các báo cáo/artifact cũ không sửa.",
              "- Cùng RTX4060 nhưng chạy khác thời điểm, không kiểm soát đầy đủ tải nền/xung/chế độ điện. Tách ASR/agent; không gán toàn bộ chênh lệch latency cho LLM hoặc chứng nhận A30.",
              "- Context 16.384, greedy, 30s/sample kể cả live ASR; startup ghi riêng. Internal 8-call cap trong thời hạn chấm theo calls/hash thực tế.",
              "- Nhóm gần trùng dựa heuristic; không có speaker IDs. Mẫu mới lấy từ train gốc, loại phần đã chạy/smoke; không tuyên bố đại diện hoàn hảo cho hidden test.",
              "- Không tự chốt model hoặc train. Kết quả giúp quyết định thứ tự ưu tiên, không bảo đảm thứ hạng sau fine-tune."]
    (args.output / "REPORT.md").write_text('\n'.join(lines) + '\n', encoding="utf-8")
    print("EXPANDED_REPORT", args.output / "REPORT.md", flush=True)
    print(json.dumps({"metrics": metrics, "comparisons": comparisons, "asr_checks": asr_checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
