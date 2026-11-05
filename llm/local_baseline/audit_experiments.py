"""Post-run TRAIN evaluation audit; never called by or fed back to model inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .data import DATA_ROOT, read_rows
from .speech import audio_paths


def audit(paths: list[Path]) -> dict:
    if "zaic_tools" in sys.modules:
        raise RuntimeError("Audit requires a fresh process without another split's runtime")
    sys.path.insert(0, str(DATA_ROOT / "train"))
    from zaic_tools import ToolEnv
    original = {row["id"]: row for row in read_rows()}
    summary = {"purpose": "Separate post-run train-only verification; no oracle feedback to agents", "runs": {}}
    for directory in paths:
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        predictions = [json.loads(line) for line in (directory / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
        traces = [json.loads(line) for line in (directory / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
        if [p["id"] for p in predictions] != manifest["ids"] or [r["id"] for r in traces] != manifest["ids"]:
            raise ValueError("Missing, duplicate, reordered or extra predictions/traces")
        replay_mismatches = []
        token_mismatches = []
        nonempty_reasoning = []
        missing_asr_clips = []
        for prediction, trace in zip(predictions, traces):
            sample_id = prediction["id"]
            if set(prediction) != {"id", "tool_calls", "state_hash"}:
                raise ValueError("Submission schema differs from official three fields")
            executed = [{"name": t["proposal"]["name"], "arguments": t["proposal"]["arguments"]}
                        for t in trace["trace"] if t.get("executed")]
            if executed != prediction["tool_calls"]:
                raise ValueError("Executed trace and submission differ: " + sample_id)
            # Separate evaluator replay, after agent process completed; not an agent reset.
            env = ToolEnv("train", sample_id, root=DATA_ROOT)
            for call in prediction["tool_calls"]:
                env.call(call["name"], call["arguments"])
            if env.get_state_hash() != prediction["state_hash"]:
                replay_mismatches.append(sample_id)
            for step in trace["trace"]:
                response = step["response"]
                if step["prompt_tokens_rendered"] != response["usage"]["prompt_tokens"]:
                    token_mismatches.append(sample_id)
                message = response["choices"][0]["message"]
                if message.get("reasoning_content") or "<think>" in (message.get("content") or ""):
                    nonempty_reasoning.append(sample_id)
            if manifest["mode"] == "asr" and trace.get("asr"):
                expected = set(audio_paths(original[sample_id]))
                if set(trace["asr"]["transcripts"]) != expected:
                    missing_asr_clips.append(sample_id)
        result = {"samples": len(predictions), "replay_hash_mismatches": replay_mismatches,
                  "token_count_mismatches": token_mismatches, "nonempty_reasoning": nonempty_reasoning,
                  "asr_coverage_mismatches": missing_asr_clips,
                  "all_checks_passed": not (replay_mismatches or token_mismatches or nonempty_reasoning or missing_asr_clips)}
        summary["runs"][str(directory)] = result
        print("AUDIT", directory.name, len(predictions), "passed", result["all_checks_passed"], flush=True)
    summary["all_checks_passed"] = all(r["all_checks_passed"] for r in summary["runs"].values())
    return summary
