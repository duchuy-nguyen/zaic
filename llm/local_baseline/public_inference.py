"""Offline public-test inference only; no labels, scoring, reference replay or training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import uuid
import wave

from .agent import infer, SYSTEM
from .backend import LocalServer, MODELS
from .data import DATA_ROOT, read_rows, catalog
from .setup_local import ROOT, TAG, sha256
from .speech import PhoWhisper, audio_paths, audio_view


SPLIT = "public_test"
PUBLIC_ROOT = DATA_ROOT / SPLIT
WEIGHTS = {"qwen35": "QWEN35_MODEL_DOWNLOAD_2026-11-04.json", "gemma": "GEMMA_MODEL_DOWNLOAD_2026-11-04.json"}
LABELS = {"qwen35": "Qwen3.5-4B", "gemma": "Gemma 4 E2B"}


def acquire_lock(path):
    lease = path.open("a+b")
    lease.seek(0)
    if not lease.read(1):
        lease.write(b"0")
        lease.flush()
    lease.seek(0)
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lease


def create_or_verify(path, content):
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Refuse to overwrite different existing output: "+str(path))
    else:
        with path.open("xb") as stream:
            stream.write(encoded)


def write_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    until = time.monotonic()+15
    while True:
        try:
            temp.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= until:
                raise
            time.sleep(.1)


def public_audio_path(relative):
    root = (PUBLIC_ROOT / "audio").resolve()
    path = (PUBLIC_ROOT / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audio must remain within public_test/audio")
    return path


class PublicPhoWhisper(PhoWhisper):
    # Only the input path changes. Same original WAV checks, decoding and model.
    def read_audio(self, relative):
        with wave.open(str(public_audio_path(relative)), "rb") as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
                raise ValueError("Expected original 16kHz mono PCM16; no silent resampling")
            frames = stream.readframes(stream.getnframes())
        values = self.np.frombuffer(frames, dtype="<i2").astype(self.np.float32)/32768.0
        if len(values) > 30*16000:
            raise ValueError("Clip exceeds short-form ASR path; no truncation")
        return values


def public_inputs():
    rows = read_rows(SPLIT)
    ids = [r["id"] for r in rows]
    if len(ids) != 634 or len(set(ids)) != 634:
        raise ValueError("Expected exactly 634 distinct public-test IDs")
    audio_hashes = {}
    for row in rows:
        if set(row) != {"id", "audio", "context", "history"}:
            raise ValueError("Unexpected public input fields; no labels/transcripts accepted")
        for turn in row.get("history", []):
            if turn["role"] == "user" and "audio_text" in turn:
                raise ValueError("Public input must not have gold transcripts")
        for relative in audio_paths(row):
            if relative in audio_hashes:
                continue
            path = public_audio_path(relative)
            with wave.open(str(path), "rb") as stream:
                if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000) or stream.getnframes() > 480000:
                    raise ValueError("Unsupported original public audio: " + relative)
            audio_hashes[relative] = sha256(path)
    return rows, audio_hashes


def verify_records(records, rows, tools, schema_validate):
    ids = [r["id"] for r in rows]
    if [r["id"] for r in records] != ids or len(set(ids)) != 634:
        raise ValueError("Submission must cover all 634 IDs exactly once in input order")
    reasoning = []
    for row, record in zip(rows, records):
        if not re.fullmatch(r"[0-9a-f]{64}", record["state_hash"]):
            raise ValueError("Invalid state hash format: " + row["id"])
        allowed = {name for app in row["context"]["installed_apps"].values() for name in app["tools"]}
        allowed.add("sys.ask_missing_info")
        bad = [v for v in schema_validate(record["tool_calls"], tools, allowed) if not v["ok"]]
        if bad:
            raise ValueError("Recorded executed call is schema-invalid: " + row["id"])
        executed = [{"name": s["proposal"]["name"], "arguments": s["proposal"]["arguments"]}
                    for s in record["trace"] if s.get("executed")]
        if executed != record["tool_calls"]:
            raise ValueError("Executed traces do not match submission calls: " + row["id"])
        for step in record["trace"]:
            response = step["response"]
            if step["prompt_tokens_rendered"] != response["usage"]["prompt_tokens"]:
                raise ValueError("Token accounting mismatch: " + row["id"])
            if step["prompt_tokens_rendered"]+512 > 16384:
                raise ValueError("Context budget exceeded: " + row["id"])
            message = response["choices"][0]["message"]
            if message.get("reasoning_content") or "<think>" in (message.get("content") or ""):
                reasoning.append(row["id"])
        if record.get("asr") and set(record["asr"]["transcripts"]) != set(audio_paths(row)):
            raise ValueError("Missing audio in ASR trace: " + row["id"])
    return {"samples": len(records), "id_schema_trace_token_checks_passed": True,
            "reasoning_samples": sorted(set(reasoning)),
            "missing_asr_samples": [r["id"] for r in records if not r.get("asr")],
            "state_hash_source": "get_state_hash from the single live session; no public replay/reset or hidden-state read"}


def finalize(directory, manifest, results, rows, tools, schema_validate):
    verification = verify_records(results, rows, tools, schema_validate)
    target = directory / "submission.jsonl"
    create_or_verify(target, "".join(json.dumps({k: r[k] for k in ("id", "tool_calls", "state_hash")}, ensure_ascii=False, allow_nan=False)+"\n" for r in results))
    loaded = [json.loads(l) for l in target.read_text(encoding="utf-8").splitlines()]
    if len(loaded) != 634 or [r["id"] for r in loaded] != manifest["ids"] or any(set(r) != {"id", "tool_calls", "state_hash"} for r in loaded):
        raise ValueError("Serialized submission failed schema/ID checks")
    if target.stat().st_size > 5*1024*1024:
        raise ValueError("Submission exceeds documented 5 MiB form limit")
    times = sorted(r["elapsed_s"] for r in results)
    statuses = dict(Counter(r["status"] for r in results))
    metrics = {"model": manifest["model"], "samples": len(results), "statuses": statuses,
               "latency": {"p50_s": statistics.median(times), "p95_s": times[math.ceil(.95*len(times))-1], "max_s": max(times)},
               "quality_score": None, "quality_note": "No public labels. Obtain actual score from BTC portal after user submits.",
               "submission_bytes": target.stat().st_size, "submission_sha256": sha256(target), "verification": verification}
    write_json(directory / "metrics.json", metrics)
    name = LABELS[manifest["model"]]
    description = (f"Zero-shot {name} Q4_K_M with PhoWhisper-small for Vietnamese audio, including user audio in history. "
                   "Offline native tool calling with the official public_test ToolEnv; one session per sample. "
                   "Greedy decoding, thinking disabled, context 16384, maximum 8 tool calls, 2 format repairs and 30 seconds per sample. "
                   "No fine-tuning, external inference service or test labels. State hashes and all executed calls come from ToolEnv.")
    create_or_verify(directory / "SOLUTION_DESCRIPTION.txt", description+"\n")
    lines = [f"# Public test — {name}", "", "Đủ 634 mẫu. File này mới được tạo local, chưa upload. Không có nhãn nên chưa biết điểm public.",
             "", f"[File để tự nộp](submission.jsonl) — {target.stat().st_size:,} bytes.",
             f"SHA256: `{metrics['submission_sha256']}`", "", "| Chỉ số | Giá trị |", "|---|---:|",
             f"| Số mẫu | {len(results)} |", f"| p50 tổng | {metrics['latency']['p50_s']:.2f}s |",
             f"| p95 tổng | {metrics['latency']['p95_s']:.2f}s |", f"| Timeout | {statuses.get('timeout',0)} |",
             f"| Guard 8 calls | {statuses.get('call_budget_exhausted',0)} |",
             f"| Mẫu phát reasoning ngoài yêu cầu | {len(verification['reasoning_samples'])} |", "",
             "Trạng thái đầy đủ: `"+json.dumps(statuses, ensure_ascii=False)+"`.", "",
             "Đã kiểm ID, JSONL 3 trường, schema/quyền tool từ context public, traces/calls, token budget và ASR coverage. Không tạo lại ToolEnv để audit/replay public; hash lấy từ phiên inference thật. Lỗi/timeout không bị bỏ hoặc chạy lại để sửa đáp án.",
             "Cùng prompt/agent/ASR và decode như baseline train; chỉ thay đường dẫn split, catalog và zaic_tools đúng public. Ngân sách tổng gồm ASR, LLM và tool. Không chứng nhận runtime A30.",
             "", "Nộp đúng file submission.jsonl, dùng mô tả trong [SOLUTION_DESCRIPTION.txt](SOLUTION_DESCRIPTION.txt) nếu phù hợp với form hiện tại. Giữ nguyên file để đối chiếu điểm BTC."]
    create_or_verify(directory / "SUMMARY.md", "\n".join(lines)+"\n")
    manifest.update(status="completed", completed_samples=len(results), finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    submission=str(target), submission_sha256=metrics["submission_sha256"])
    write_json(directory / "run_manifest.json", manifest)
    return metrics


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WEIGHTS, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    rows, audio_hashes = public_inputs()
    tools = catalog(SPLIT)
    ids = [r["id"] for r in rows]
    weight_manifest = json.loads((ROOT / WEIGHTS[args.model]).read_text(encoding="utf-8"))
    weight = next(w for w in weight_manifest["models"] if w["filename"] == MODELS[args.model])
    if sha256(ROOT / weight["filename"]) != weight["expected_sha256"]:
        raise ValueError("Weights fingerprint mismatch")
    critical = ["agent.py", "backend.py", "data.py", "speech.py", "public_inference.py"]
    common = {"model": args.model, "mode": "asr", "split": SPLIT, "ids": ids,
              "input_sha256": sha256(PUBLIC_ROOT / "public_test.jsonl"), "catalog_sha256": sha256(PUBLIC_ROOT / "public_test_tools_catalog.json"),
              "audio_sha256": audio_hashes, "code_sha256": {name: sha256(Path(__file__).parent / name) for name in critical},
              "runtime_py_sha256": {str(p.relative_to(PUBLIC_ROOT / "zaic_tools")): sha256(p) for p in sorted((PUBLIC_ROOT / "zaic_tools").rglob("*.py"))},
              "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), "runtime_tag": TAG, "weight": weight,
              "seed": 20261103, "temperature": 0, "thinking": False, "context": 16384, "max_new_tokens": 512,
              "max_calls": 8, "max_repairs": 2, "deadline_s": 30, "fine_tuned": False,
              "scope": "Public inference only. No labels/references/training/scoring or upload.",
              "normalization": "NFC and whitespace only, no entity correction/gold fallback"}
    directory = args.resume.resolve() if args.resume else ROOT / "runs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+f"_public_{args.model}_"+uuid.uuid4().hex[:6])
    if args.resume:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        for k, v in common.items():
            if manifest.get(k) != v:
                raise ValueError("Cannot resume; changed configuration: "+k)
        if manifest["status"] == "completed":
            print("PUBLIC_ALREADY_COMPLETE", manifest["submission"], flush=True)
            return
    else:
        directory.mkdir(exist_ok=False)
        manifest = {**common, "status": "preparing", "completed_samples": 0, "startup_sessions": [],
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(directory / "run_manifest.json", manifest)
        snapshot = directory / "code_snapshot"
        snapshot.mkdir()
        for name in critical:
            (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
    lease = acquire_lock(ROOT / "runs/public_inference.lock")
    print("PUBLIC_RUN_DIR", directory, flush=True)
    records = []
    trace = directory / "traces.jsonl"
    if trace.exists():
        records = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
        if [r["id"] for r in records] != ids[:len(records)]:
            raise ValueError("Checkpoint is not an exact committed prefix")
    inflight = directory / "inflight.json"
    if inflight.exists():
        unfinished = json.loads(inflight.read_text(encoding="utf-8"))["id"]
        if unfinished not in {r["id"] for r in records}:
            raise RuntimeError("Uncommitted public session exists for "+unfinished+"; refuse automatic reset/rerun. Preserve artifacts and investigate.")
        inflight.unlink()  # Its result was flushed before the interruption.
    if "zaic_tools" in sys.modules:
        raise RuntimeError("Fresh process required for the public runtime")
    sys.path.insert(0, str(PUBLIC_ROOT))
    from zaic_tools import ToolEnv, schema_validate
    if len(records) == len(ids):
        metrics = finalize(directory, manifest, records, rows, tools, schema_validate)
        print("PUBLIC_COMPLETE", directory, json.dumps(metrics, ensure_ascii=False), flush=True)
        return
    startup = time.monotonic()
    asr = PublicPhoWhisper()
    if manifest.get("asr") and manifest["asr"] != asr.config:
        raise ValueError("ASR configuration changed")
    manifest["asr"] = asr.config
    session_dir = directory / ("server-session-"+str(len(manifest["startup_sessions"])+1))
    with LocalServer(args.model, session_dir) as backend:
        manifest["startup_sessions"].append({"seconds": time.monotonic()-startup, "server_command": backend.public_command()})
        manifest.update(status="running")
        write_json(directory / "run_manifest.json", manifest)
        with trace.open("a", encoding="utf-8") as output:
            for row in rows[len(records):]:
                write_json(inflight, {"id": row["id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "policy": "Do not reset/retry this public session if interrupted before trace commit"})
                started = time.monotonic()
                env = ToolEnv(SPLIT, row["id"], root=DATA_ROOT)  # Exactly one per sample, no probes/replay.
                details = None
                try:
                    transcripts, details = asr.transcribe(audio_paths(row), started+30)
                    view = audio_view(row, transcripts)
                    remaining = 30-(time.monotonic()-started)
                    if remaining <= 0:
                        raise TimeoutError("ASR/input preparation exhausted sample deadline")
                    result = infer(view, tools, env, backend, seconds=remaining)
                except Exception as error:
                    result = {"id": row["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
                              "status": "timeout" if isinstance(error, TimeoutError) else "asr_error",
                              "error": repr(error), "trace": [], "repairs": 0}
                result["llm_and_agent_s"] = result.pop("elapsed_s", 0.)
                result["elapsed_s"] = time.monotonic()-started
                result["asr"] = details
                if result["elapsed_s"] > 30:
                    result["status"] = "timeout"
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
                output.flush()
                records.append(result)
                inflight.unlink()
                manifest["completed_samples"] = len(records)
                write_json(directory / "run_manifest.json", manifest)
                print(f"[{len(records)}/634] {row['id']} {result['status']} calls={len(result['tool_calls'])} total={result['elapsed_s']:.2f}s", flush=True)
    metrics = finalize(directory, manifest, records, rows, tools, schema_validate)
    print("PUBLIC_COMPLETE", directory, json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
