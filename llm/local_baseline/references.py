"""Separate TRAIN-only annotation job. Never imported by agent inference."""
from __future__ import annotations

import json
import sys

from .data import DATA_ROOT, ROOT, read_rows, catalog
from .setup_local import sha256
from .scoring import score

PATH = ROOT / "data" / "references" / "train-v1.json"


def build() -> dict:
    source_hash = sha256(DATA_ROOT / "train" / "train.jsonl")
    if PATH.exists():
        saved = json.loads(PATH.read_text(encoding="utf-8"))
        if saved["train_jsonl_sha256"] != source_hash:
            raise RuntimeError("Reference dataset changed")
        return saved
    if "zaic_tools" in sys.modules:
        raise RuntimeError("Run reference preparation in a separate process")
    sys.path.insert(0, str(DATA_ROOT / "train"))
    from zaic_tools import ToolEnv, canon
    tools = catalog()
    refs, mismatches = {}, []
    for row in read_rows():
        # This gold replay is a separate train annotation job, not an agent retry/reset.
        env = ToolEnv("train", row["id"], root=DATA_ROOT)
        for call in row["label"]["tool_calls"]:
            result = env.call(call["name"], call["arguments"])
            if not result.get("ok"):
                raise ValueError(f"Gold schema failure: {row['id']}")
        state = env.get_state()  # Allowed only here, train annotation/evaluation.
        partial = row["label"]["final_state"]
        expected = canon(partial)
        actual = canon({app: state[app] for app in partial})
        if expected != actual:
            mismatches.append(row["id"])
        reference = {"tool_calls": row["label"]["tool_calls"], "state_hash": env.get_state_hash(),
                     "partial_label_consistent": expected == actual}
        self_score = score({"tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}, reference, tools)
        if self_score["score_spec_local"] != 100:
            raise ValueError("Gold replay self-score was not 100")
        refs[row["id"]] = reference
    saved = {"version": "gold-replay-train-v1", "train_jsonl_sha256": source_hash,
             "partial_label_mismatches": mismatches, "references": refs}
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with PATH.open("x", encoding="utf-8") as stream:
        json.dump(saved, stream, ensure_ascii=False, indent=2)
    return saved


if __name__ == "__main__":
    data = build()
    print(json.dumps({"references": len(data["references"]), "partial_label_mismatches": data["partial_label_mismatches"]}, ensure_ascii=False))
