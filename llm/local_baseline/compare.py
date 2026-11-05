"""Compare two completed runs only if inputs and important settings match."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First run directory")
    parser.add_argument("right", type=Path, help="Second run directory")
    parser.add_argument("--output", type=Path, help="New Markdown file; never overwrites an existing report")
    args = parser.parse_args()
    left, right = [json.loads((p / "run_manifest.json").read_text(encoding="utf-8")) for p in (args.left, args.right)]
    fields = ["ids", "split_sha256", "mode", "fine_tuned", "context", "max_new_tokens", "max_calls", "max_repairs", "seconds", "seed", "temperature", "thinking", "scorer", "scorer_assumptions", "system_prompt_sha256", "runtime_tag"]
    mismatches = [key for key in fields if left.get(key) != right.get(key)]
    if mismatches:
        raise SystemExit("Not a controlled comparison; settings differ: " + ", ".join(mismatches))
    critical_code = ["agent.py", "backend.py", "data.py", "scoring.py", "run.py"]
    changed = [key for key in critical_code if left.get("code_sha256", {}).get(key) != right.get("code_sha256", {}).get(key)]
    if changed:
        raise SystemExit("Inference/scoring code differs between runs: " + ", ".join(changed))
    scores = []
    for path in (args.left, args.right):
        records = [json.loads(line) for line in (path / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
        if [r["id"] for r in records] != left["ids"]:
            raise SystemExit("Score records differ from manifest IDs/order")
        scores.append({r["id"]: r for r in records})
    metrics = [json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in (args.left, args.right)]
    wins = sum(scores[1][i]["score_spec_local"] > scores[0][i]["score_spec_local"] for i in left["ids"])
    losses = sum(scores[1][i]["score_spec_local"] < scores[0][i]["score_spec_local"] for i in left["ids"])
    lines = ["# So sánh hai lần chạy local", "", f"Mẫu: {len(left['ids'])}; partition: {left['partition']}.",
             "**Đây là smoke trên TRAIN_FIT: chỉ kiểm tra tích hợp, không kết luận model thắng trên validation.**" if left["partition"] == "smoke" else "Kết quả trên VAL_MAIN cố định; cần xem coverage và uncertainty trước kết luận.",
             "Điểm là local-spec-v1, chưa được BTC chứng nhận. Thời gian là gold-text trên máy local, không bao gồm ASR.",
             "", "| Model | Điểm local /100 | Tool F1 | State match | p50 text | p95 text | Status |", "|---|---|---|---|---|---|---|"]
    for manifest, value in zip((left, right), metrics):
        latency = value["latency_text_only"]
        lines.append(f"| {manifest['model']} | {value['score_spec_local']:.2f} | {value['tool_f1']:.3f} | {value['state_accuracy']:.3f} | {latency['p50_s']:.2f}s | {latency['p95_s']:.2f}s | {value['statuses']} |")
    lines += ["", f"So với {left['model']}, {right['model']}: hơn {wins}, kém {losses}, bằng {len(left['ids'])-wins-losses} mẫu.",
              "", "| ID | Điểm trái | Điểm phải |", "|---|---|---|"]
    for sample_id in left["ids"]:
        lines.append(f"| {sample_id} | {scores[0][sample_id]['score_spec_local']:.2f} | {scores[1][sample_id]['score_spec_local']:.2f} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report)


if __name__ == "__main__":
    main()
