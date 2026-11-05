"""Freeze 600 new train examples before inference; never modify the old split."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from .data import MANIFEST, prepare, read_rows, taxonomy
from .setup_local import ROOT, sha256

SEED = 20261104
COUNT = 600
PATH = ROOT / "data/manifests/expanded-zeroshot-600-v1.json"


def validate_selection(selection, split):
    ids = selection["ids"]
    if len(ids) != COUNT or len(set(ids)) != COUNT:
        raise ValueError("Expected exactly 600 distinct selected IDs")
    if selection["base_split_sha256"] != sha256(MANIFEST):
        raise ValueError("Base split changed")
    if not set(ids) <= set(split["train_fit"]):
        raise ValueError("Expanded samples must come from original TRAIN_FIT")
    excluded = set(selection["excluded_ids"]) | set(split["val_main"]) | set(split["smoke_train_fit"])
    blocked_groups = {split["group_ids"][i] for i in excluded}
    if {split["group_ids"][i] for i in ids} & blocked_groups:
        raise ValueError("Previously seen sample or near-duplicate group selected")
    groups = defaultdict(set)
    for sample_id in split["train_fit"]:
        groups[split["group_ids"][sample_id]].add(sample_id)
    selected = set(ids)
    if any(group & selected and not group <= selected for group in groups.values()):
        raise ValueError("Selected duplicate group is incomplete")
    return ids


def create():
    split = prepare()
    if PATH.exists():
        selection = json.loads(PATH.read_text(encoding="utf-8"))
        validate_selection(selection, split)
        return selection
    seen = set(split["val_main"]) | set(split["smoke_train_fit"])
    inspected = []
    for path in sorted((ROOT / "runs").glob("*/run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        seen.update(manifest.get("ids", []))
        inspected.append({"path": str(path), "sha256": sha256(path)})
    blocked_groups = {split["group_ids"][i] for i in seen}
    groups = defaultdict(list)
    for sample_id in sorted(split["train_fit"]):
        group = split["group_ids"][sample_id]
        if group not in blocked_groups:
            groups[group].append(sample_id)
    candidates = list(groups.values())
    random.Random(SEED).shuffle(candidates)
    ids = []
    for group in candidates:
        if len(ids) + len(group) <= COUNT:
            ids.extend(group)
        if len(ids) == COUNT:
            break
    ids.sort()
    rows = {r["id"]: r for r in read_rows()}
    selection = {"version": "expanded-zeroshot-600-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "seed": SEED, "base_split_sha256": sha256(MANIFEST), "ids": ids,
                 "excluded_ids": sorted(seen), "prior_manifests": inspected,
                 "selection": "Uniform seeded shuffle of eligible duplicate groups; no score-based selection or early stopping",
                 "future_training": "All original train samples may be used for later fine-tuning; not a permanent holdout",
                 "distribution": {"samples": len(ids), "call_counts": dict(Counter(taxonomy(rows[i])["calls"] for i in ids)),
                                  "ask": sum(taxonomy(rows[i])["ask"] for i in ids),
                                  "with_history": sum(bool(rows[i].get("history")) for i in ids),
                                  "long_history_ge10": sum(len(rows[i].get("history", [])) >= 10 for i in ids)}}
    validate_selection(selection, split)
    with PATH.open("x", encoding="utf-8") as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2)
    return selection


if __name__ == "__main__":
    print(json.dumps({"path": str(PATH), "distribution": create()["distribution"]}, ensure_ascii=False))
