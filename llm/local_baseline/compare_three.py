"""Compare Gemma's additional runs against preserved Qwen/Spark artifacts."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import statistics

from .analyze_experiments import load, paired
from .audit_experiments import audit
from .data import MANIFEST
from .setup_local import ROOT


def verify_sources(old, new):
    """Accept only the exact model registry / manifest plumbing additions."""
    changes = {}
    for filename in old["manifest"]["code_sha256"]:
        before = (Path(old["path"]) / "code_snapshot" / filename).read_text(encoding="utf-8")
        after = (Path(new["path"]) / "code_snapshot" / filename).read_text(encoding="utf-8")
        if hashlib.sha256(before.encode()).digest() == hashlib.sha256(after.encode()).digest():
            continue
        normalized = after
        if filename == "backend.py":
            normalized = normalized.replace('    "gemma": "gemma-4-E2B-it-Q4_K_M.gguf",\n', '')
            normalized = normalized.replace('"add_special": self.model == "gemma"', '"add_special": False')
        elif filename == "experiment.py":
            normalized = normalized.replace('    parser.add_argument("--weights-manifest", type=Path, default=ROOT / "MODEL_DOWNLOADS_2026-11-04.json")\n', '')
            normalized = normalized.replace('json.loads(args.weights_manifest.read_text(encoding="utf-8"))["models"]',
                                            'json.loads((ROOT / "MODEL_DOWNLOADS_2026-11-04.json").read_text(encoding="utf-8"))["models"]')
        if normalized != before:
            raise ValueError("Unexpected inference/scoring code change: " + filename)
        changes[filename] = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True)))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-gold", type=Path, required=True)
    parser.add_argument("--gemma-asr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((ROOT / "runs/validation_campaign_20261104T104233Z/campaign.json").read_text(encoding="utf-8"))
    runs = {f"{job['model']}-{job['mode']}": load(Path(job["run_dir"])) for job in previous["jobs"]}
    runs.update({"gemma-gold": load(args.gemma_gold), "gemma-asr": load(args.gemma_asr)})
    split = json.loads(MANIFEST.read_text(encoding="utf-8"))
    shared = ["ids", "split_sha256", "catalog_sha256", "runtime_py_sha256", "reference_sha256",
              "prompt_sha256", "scorer", "scorer_assumptions", "runtime_tag", "seed", "thinking", "temperature",
              "context", "max_new_tokens", "max_calls", "max_repairs", "deadline_s", "normalization", "fine_tuned"]
    base = runs["qwen-gold"]
    differences = {}
    for key, run in runs.items():
        if run["manifest"]["ids"] != split["val_main"]:
            raise ValueError("Not the full frozen validation split: " + key)
        for field in shared:
            if run["manifest"].get(field) != base["manifest"].get(field):
                raise ValueError(f"Uncontrolled configuration difference {key}: {field}")
        differences[key] = verify_sources(base, run)
    asr = runs["qwen-asr"]["manifest"]["asr"]
    for model in ("spark", "gemma"):
        if runs[f"{model}-asr"]["manifest"]["asr"] != asr:
            raise ValueError("ASR configuration differs")
    comparisons = {f"{mode}_gemma_minus_{model}": paired(runs[f"{model}-{mode}"], runs[f"gemma-{mode}"], split["group_ids"])
                   for mode in ("gold", "asr") for model in ("qwen", "spark")}
    transcripts = {}
    for model in ("spark", "gemma"):
        equal, different, missing = [], [], []
        for sample_id in split["val_main"]:
            a = runs["qwen-asr"]["scores"][sample_id]["asr_transcript_sha256"]
            b = runs[f"{model}-asr"]["scores"][sample_id]["asr_transcript_sha256"]
            (missing if a is None or b is None else equal if a == b else different).append(sample_id)
        transcripts[model] = {"equal": len(equal), "different": different, "missing": missing}
    verification = audit([args.gemma_gold, args.gemma_asr])
    args.output.mkdir(parents=True, exist_ok=False)
    analysis = {"metrics": {k: r["metrics"] for k, r in runs.items()}, "comparisons": comparisons,
                "asr_transcript_comparison_to_qwen": transcripts, "allowed_source_differences": differences,
                "run_paths": {k: r["path"] for k, r in runs.items()}, "audit": verification,
                "budget_stop_corrections": {k: r["budget_stop_corrections"] for k, r in runs.items()},
                "not_official_score": True, "model_selection": "Deferred to user; no fine-tuning"}
    (args.output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# So sánh zero-shot local: Qwen, Spark, Gemma 4 E2B", "",
             "Mỗi model: 309 mẫu text chuẩn + 309 mẫu audio qua PhoWhisper-small live. Cùng split, prompt, scorer và cấu hình giới hạn.",
             "Điểm local bám công thức BTC, không phải điểm leaderboard. Không fine-tune, few-shot hay RAG.", "",
             "| Model | Text /100 | Audio /100 | Audio p50 | Audio p95 | Audio lỗi/timeout |",
             "|---|---|---|---|---|---|"]
    for model in ("qwen", "spark", "gemma"):
        text, audio = runs[f"{model}-gold"]["metrics"], runs[f"{model}-asr"]["metrics"]
        lines.append(f"| {model} | {text['score_spec_local']:.4f} | {audio['score_spec_local']:.4f} | {audio['latency']['p50_s']:.2f}s | {audio['latency']['p95_s']:.2f}s | {audio['failed_samples_counted_zero']} |")
    lines += ["", "## Tách thời gian ASR và agent", "",
              "| Model | ASR p50 | ASR p95 | LLM + agent p50 | LLM + agent p95 |",
              "|---|---|---|---|---|"]
    for model in ("qwen", "spark", "gemma"):
        run = runs[f"{model}-asr"]
        times = sorted(row["llm_and_agent_s"] for row in run["scores"].values())
        import math
        asr_time = run["metrics"]["asr_latency"]
        lines.append(f"| {model} | {asr_time['p50_s']:.2f}s | {asr_time['p95_s']:.2f}s | {statistics.median(times):.2f}s | {times[math.ceil(.95*len(times))-1]:.2f}s |")
    lines += ["", "ASR cũng có thể nhanh/chậm khác nhau giữa các lượt dù cùng cấu hình. Các lượt chạy khác thời điểm, không kiểm soát đầy đủ xung GPU, chế độ điện và tải máy. Không quy toàn bộ chênh lệch end-to-end cho kiến trúc LLM; latency là quan sát local, không phải benchmark tốc độ được kiểm soát hoàn toàn.",
              "Các percentile từng giai đoạn không cộng trực tiếp thành percentile tổng.",
              "", "## Chênh lệch trên cùng mẫu", "", "| Phép so | Delta | CI95% bootstrap theo group |", "|---|---|---|"]
    for name, value in comparisons.items():
        low, high = value["ci95_group_bootstrap"]
        lines.append(f"| {name} | {value['delta_right_minus_left']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines += ["", "## Kiểm tra và giới hạn", "",
              f"- Audit replay/hash, calls/traces, token counting, reasoning và ASR coverage Gemma: {'PASS' if verification['all_checks_passed'] else 'FAIL — xem analysis.json'}.",
              "- ASR hashes so với Qwen: " + json.dumps(transcripts, ensure_ascii=False),
              "- Qwen/Spark dùng artifacts gốc; Gemma thêm tên model, manifest weights và đếm BOS mà server tự thêm. Comparator kiểm đúng các thay đổi đó; agent/prompt/scorer/ASR giữ nguyên.",
              "- Cùng Q4_K_M nhưng tokenizer, native template và kiến trúc riêng. Thời điểm chạy khác nhau, nhiệt độ/tải máy có thể ảnh hưởng latency.",
              "- Startup không tính trong 30 giây mỗi sample; chưa chứng nhận phần cứng A30 của BTC.",
              "- Giữ lỗi/timeout trong đủ 309 mẫu. Guard 8 calls trong thời hạn được chấm theo calls/hash thực tế; không sửa artifacts gốc.",
              "- Fuzzy tokenizer/exact-key corner cases là assumptions local-spec-v1; đây là validation chọn model, không phải test cuối chưa từng được xem.",
              "- Gemma dùng nhánh ngôn ngữ với ASR ngoài, không thử audio native. Nhãn E2B là effective parameters; eligibility cần tính đúng phần LLM/tổng model thực tế.",
              "- Chưa tự chọn model hoặc fine-tune.", "", "## Artifacts", ""]
    for key, run in runs.items():
        lines.append(f"- {key}: [{Path(run['path']).name}]({Path(run['path']).as_posix()}/SUMMARY.md)")
    (args.output / "REPORT.md").write_text('\n'.join(lines) + '\n', encoding="utf-8")
    print("THREE_MODEL_REPORT", args.output / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()
