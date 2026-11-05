"""Two separate local public-test submissions, Gemma first then Qwen3.5-4B."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .public_inference import ROOT, LABELS, write_json, acquire_lock, create_or_verify
from .setup_local import sha256


def finish(directory, manifest):
    runs = {j["model"]: Path(j["run_dir"]) for j in manifest["jobs"]}
    records = {m: [json.loads(l) for l in (p / "traces.jsonl").read_text(encoding="utf-8").splitlines()] for m, p in runs.items()}
    metrics = {m: json.loads((p / "metrics.json").read_text(encoding="utf-8")) for m, p in runs.items()}
    manifests = {m: json.loads((p / "run_manifest.json").read_text(encoding="utf-8")) for m, p in runs.items()}
    fields = ["ids", "input_sha256", "catalog_sha256", "audio_sha256", "code_sha256", "runtime_py_sha256",
              "prompt_sha256", "runtime_tag", "seed", "temperature", "thinking", "context", "max_new_tokens",
              "max_calls", "max_repairs", "deadline_s", "fine_tuned", "asr", "normalization"]
    for field in fields:
        if manifests["gemma"].get(field) != manifests["qwen35"].get(field):
            raise ValueError("Changed public configuration between models: "+field)
    same, different, missing = [], [], []
    for a, b in zip(records["gemma"], records["qwen35"]):
        if a["id"] != b["id"]:
            raise ValueError("Unpaired public IDs")
        if not a.get("asr") or not b.get("asr"):
            missing.append(a["id"])
        elif a["asr"]["transcript_sha256"] == b["asr"]["transcript_sha256"]:
            same.append(a["id"])
        else:
            different.append(a["id"])
    output = directory / "submissions"
    output.mkdir(exist_ok=True)
    files = {}
    for model, run in runs.items():
        filename = "submission_qwen35_4b.jsonl" if model == "qwen35" else "submission_gemma4_e2b.jsonl"
        target = output / filename
        source = run / "submission.jsonl"
        if sha256(source) != manifests[model]["submission_sha256"]:
            raise ValueError("Submission changed after generation")
        create_or_verify(target, source.read_text(encoding="utf-8"))
        if sha256(target) != sha256(source):
            raise ValueError("Submission copy differs")
        files[model] = str(target)
        description = (run / "SOLUTION_DESCRIPTION.txt").read_text(encoding="utf-8")
        create_or_verify(output / (model+"_solution_description.txt"), description)
    verification = {"public_ids_per_model": 634, "asr_same": len(same), "asr_different_ids": different,
                    "asr_missing_ids": missing, "submissions": files, "metrics": metrics,
                    "not_uploaded": True, "quality_score": None}
    write_json(directory / "verification.json", verification)
    lines = ["# Hai bài public test để tự nộp", "", "Mỗi model có đủ 634 mẫu public test; chưa fine-tune và chưa upload. **Chưa có điểm chất lượng** vì tập public không có nhãn. Điểm sẽ do BTC trả sau khi bạn nộp.",
             "", "| Model | File nộp | Mẫu | p50 tổng | p95 tổng | Timeout | Lỗi khác |", "|---|---|---:|---:|---:|---:|---:|"]
    for model in ("qwen35", "gemma"):
        m = metrics[model]
        errors = sum(n for status, n in m["statuses"].items() if status not in ("finished", "asked_missing_info", "call_budget_exhausted", "timeout"))
        lines.append(f"| {LABELS[model]} | [{Path(files[model]).name}]({Path(files[model]).as_posix()}) | 634 | {m['latency']['p50_s']:.2f}s | {m['latency']['p95_s']:.2f}s | {m['statuses'].get('timeout',0)} | {errors} |")
    lines += ["", "## Cách dùng", "",
              "1. Trên portal LifeMate, chọn nộp kết quả **public test**.",
              "2. Nộp từng file JSONL riêng; tên file đã ghi rõ model. Không nộp traces, ZIP dataset hoặc file báo cáo.",
              "3. Mô tả tham khảo nằm cạnh file nộp, trong `*_solution_description.txt`; mỗi mô tả dưới 1.000 ký tự.",
              "4. Lưu điểm BTC kèm đúng model/tên file/thời điểm để so sánh. Kiểm quota trên portal trước khi nộp hai lượt.",
              "", "## Kiểm tra", "",
              "- Đúng 634 ID duy nhất, mỗi dòng chỉ có id/tool_calls/state_hash, UTF-8 và dưới giới hạn form đã ghi nhận 5 MiB.",
              "- Calls/hash lấy từ đúng một ToolEnv public cho mỗi mẫu/model. Không đọc trực tiếp environment, không get_state/replay/reset public để dò đáp án; audit chỉ dùng input/schema và trace đã ghi.",
              f"- Transcript hash giống giữa hai model: {len(same)}/634; khác {len(different)}, thiếu {len(missing)}. Chi tiết trong verification.json.",
              "- Weights Q4_K_M, PhoWhisper-small live cho current/history, prompt/agent và giới hạn giữ như baseline. 30s/sample gồm ASR+LLM+tool, 8 calls/2 repairs, context 16384, greedy/seed cố định.",
              "- Lỗi/timeout giữ nguyên trong file, không rerun chọn lọc hoặc lọc mẫu. Reasoning dù đã yêu cầu tắt được ghi rõ trong SUMMARY.md từng model, không che giấu hoặc bỏ mẫu.",
              "- Chưa phải điểm leaderboard và chưa kiểm chứng runtime trên A30. Hai lượt khác thời điểm/tải máy. Không tạo nhãn/tune trên public test.",
              "- Lượt chẩn đoán text chuẩn đang paused_by_user sau 42/1545 mẫu và không tự resume trong campaign này.", ""]
    create_or_verify(directory / "READY_TO_SUBMIT.md", "\n".join(lines)+"\n")
    return files


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        directory = args.resume.resolve()
        manifest = json.loads((directory / "campaign.json").read_text(encoding="utf-8"))
        if manifest["status"] == "completed":
            print("PUBLIC_CAMPAIGN_ALREADY_COMPLETE", manifest["report"], flush=True)
            return
    else:
        directory = ROOT / "runs" / datetime.now(timezone.utc).strftime("public_two_campaign_%Y%m%dT%H%M%SZ")
        directory.mkdir(exist_ok=False)
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
                    "scope": "Generate two full public_test submissions locally; user submits; no training/upload",
                    "jobs": [{"model": m, "expected_samples": 634, "status": "pending", "sessions": 0} for m in ("gemma", "qwen35")]}
    lease = acquire_lock(ROOT / "runs/public_campaign.lock")
    manifest["status"] = "running"
    write_json(directory / "campaign.json", manifest)
    print("PUBLIC_CAMPAIGN_DIR", directory, flush=True)
    python = str(ROOT / ".runtime/asr-venv/Scripts/python.exe")
    try:
        for job in manifest["jobs"]:
            if job["status"] == "completed":
                continue
            run = Path(job["run_dir"]) if job.get("run_dir") else None
            done = run and (run / "metrics.json").is_file() and json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))["status"] == "completed"
            if not done:
                command = [python, "-u", "-m", "local_baseline.public_inference", "--model", job["model"]]
                if run:
                    command += ["--resume", str(run)]
                job.update(status="running", sessions=job["sessions"]+1, command=command)
                write_json(directory / "campaign.json", manifest)
                with (directory / f"{job['model']}_session{job['sessions']}.log").open("x", encoding="utf-8") as log:
                    with subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace",
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            print(line, end="", flush=True)
                            if line.startswith("PUBLIC_RUN_DIR "):
                                job["run_dir"] = line[len("PUBLIC_RUN_DIR "):].strip()
                                write_json(directory / "campaign.json", manifest)
                        code = process.wait()
                if code:
                    raise RuntimeError(f"{job['model']} public inference exited {code}; preserve committed results, no selective rerun")
                run = Path(job["run_dir"])
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if metrics["samples"] != 634:
                raise ValueError("Incomplete public submission")
            job.update(status="completed", metrics=metrics)
            write_json(directory / "campaign.json", manifest)
        manifest.update(status="verifying")
        write_json(directory / "campaign.json", manifest)
        files = finish(directory, manifest)
        manifest.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        submissions=files, report=str(directory / "READY_TO_SUBMIT.md"))
        manifest.pop("error", None)
        write_json(directory / "campaign.json", manifest)
        print("PUBLIC_CAMPAIGN_COMPLETE", manifest["report"], flush=True)
    except BaseException as error:
        manifest.update(status="interrupted_or_failed", error=repr(error))
        write_json(directory / "campaign.json", manifest)
        raise


if __name__ == "__main__":
    main()
