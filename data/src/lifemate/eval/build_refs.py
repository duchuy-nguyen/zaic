"""Tạo input sạch + references (train-only) + danh sách ID theo split, TÁCH RIÊNG nhau (OPS CHECKLIST mục DATA).

  inputs_<split>.jsonl      id, audio, audio_text, transcript_source, context, history — KHÔNG có label/state
  references_<split>.jsonl  id, tool_calls (gold), state_hash (replay gold trên ToolEnv train), behavior — chỉ cho evaluator
  expected_ids_<split>.jsonl {"id": ...}  (định dạng PREFLIGHT.md của OPS) — chỉ ID, để được trên git

split: fit | val (theo manifests/split_v1.json) | public_test (chỉ expected_ids, nếu có official/public_test).
state_hash gold = ToolEnv("train", id).get_state_hash() SAU khi gọi lần lượt gold tool_calls (đúng cách BTC replay).

  python -m src.lifemate.eval.build_refs --splits val fit --out eval_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..synthetic.common import MANIFEST, OFFICIAL, load_jsonl
from ..synthetic.env_runner import train_env
from .scorer import VERSION, behavior

TRANSCRIPT_SOURCE = "gold_audio_text_train_only"  # public/private phải dùng ASR, không fallback gold


def clean_input(s: dict) -> dict:
    return {"id": s["id"], "audio": s["audio"], "audio_text": s.get("audio_text"),
            "transcript_source": TRANSCRIPT_SOURCE, "context": s["context"], "history": s.get("history", [])}


def gold_reference(s: dict) -> dict:
    env = train_env(s["id"])
    res = [env.call(c["name"], c["arguments"]) for c in s["label"]["tool_calls"]]
    if res != s["label"]["tool_results"]:
        raise RuntimeError(f"{s['id']}: replay gold khác label.tool_results")
    return {"id": s["id"], "tool_calls": s["label"]["tool_calls"], "state_hash": env.get_state_hash(),
            "behavior": behavior(s["label"]["tool_calls"])}


def write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["fit", "val", "public_test"])
    ap.add_argument("--out", default="eval_v1")
    ap.add_argument("--ids-out", default=None, help="thư mục ghi expected_ids_*.jsonl (vd manifests); mặc định = --out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids_out = Path(a.ids_out) if a.ids_out else out
    ids_out.mkdir(parents=True, exist_ok=True)
    split = json.loads((MANIFEST / "split_v1.json").read_text(encoding="utf-8"))
    summary = {"scorer_version": VERSION, "split_name": split["version"],
               "split_manifest_sha256": hashlib.sha256((MANIFEST / "split_v1.json").read_bytes()).hexdigest(),
               "transcript_source": TRANSCRIPT_SOURCE, "files": {}}
    train = None
    for sp in a.splits:
        if sp == "public_test":
            p = OFFICIAL / "public_test" / "public_test.jsonl"
            if not p.exists():
                print("bỏ qua public_test: không thấy", p)
                continue
            ids = [r["id"] for r in load_jsonl(p)]
            summary["files"][f"expected_ids_{sp}.jsonl"] = write_jsonl(ids_out / f"expected_ids_{sp}.jsonl", [{"id": i} for i in ids])
            summary[f"n_{sp}"] = len(ids)
            continue
        train = train or {s["id"]: s for s in load_jsonl(OFFICIAL / "train" / "train.jsonl")}
        ids = sorted(split[sp])
        rows = [train[i] for i in ids]
        summary["files"][f"inputs_{sp}.jsonl"] = write_jsonl(out / f"inputs_{sp}.jsonl", [clean_input(s) for s in rows])
        refs = [gold_reference(s) for s in rows]
        summary["files"][f"references_{sp}.jsonl"] = write_jsonl(out / f"references_{sp}.jsonl", refs)
        summary["files"][f"expected_ids_{sp}.jsonl"] = write_jsonl(ids_out / f"expected_ids_{sp}.jsonl", [{"id": i} for i in ids])
        summary[f"n_{sp}"] = len(ids)
        summary[f"behavior_{sp}"] = {b: sum(r["behavior"] == b for r in refs) for b in ("action", "ask", "empty")}
    (out / "eval_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
