"""Dataset loading and frozen train-only grouped split. Never reads environment files."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from .setup_local import ROOT, sha256

DATA_ROOT = ROOT / "data" / "lifemate-v3.2.0" / "train+public_test"
MANIFEST = ROOT / "data" / "manifests" / "split-v1.json"


def read_rows(split: str = "train") -> list[dict]:
    with (DATA_ROOT / split / f"{split}.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def catalog(split: str = "train") -> dict[str, dict]:
    data = json.loads((DATA_ROOT / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
    return {tool["name"]: tool for app in data["apps"] for tool in app.get("tools", [])}


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).lower().split())


def agent_view(row: dict) -> dict:
    """Gold-text train mode only. Deliberate allowlist excludes label/final_state."""
    if "audio_text" not in row:
        raise ValueError("This runner is gold-text TRAIN-only; public/private need an ASR provider")
    history = []
    for turn in row.get("history", []):
        if turn["role"] == "user":
            if "audio_text" not in turn:
                raise ValueError("Missing gold history transcript; no silent fallback")
            history.append({"role": "user", "text": turn["audio_text"]})
        else:
            history.append({k: turn[k] for k in ("role", "text", "tool_calls", "tool_results") if k in turn})
    return {"id": row["id"], "latest_text": row["audio_text"], "context": row["context"], "history": history}


def taxonomy(row: dict) -> dict:
    calls = row["label"]["tool_calls"]
    return {"calls": len(calls), "ask": any(c["name"] == "sys.ask_missing_info" for c in calls),
            "history_turns": len(row.get("history", []))}


def prepare(seed: int = 20261103) -> dict:
    source_hash = sha256(DATA_ROOT / "train" / "train.jsonl")
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing["train_jsonl_sha256"] != source_hash:
            raise RuntimeError("Dataset differs from frozen split; create an explicit new version")
        return existing
    rows = read_rows()
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate sample IDs")
    parent = {sample_id: sample_id for sample_id in by_id}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    seen = {}
    current_audio_hashes = {}
    by_skeleton = defaultdict(list)
    for row in rows:
        sample_id = row["id"]
        latest = normalize(row["audio_text"])
        dialogue = [normalize(t["audio_text"]) for t in row.get("history", []) if t["role"] == "user"] + [latest]
        audio_hash = sha256(DATA_ROOT / "train" / row["audio"])
        current_audio_hashes[sample_id] = audio_hash
        for key in [("latest", latest), ("dialogue", tuple(dialogue)), ("current_audio", audio_hash)]:
            if key in seen:
                union(sample_id, seen[key])
            else:
                seen[key] = sample_id
        skeleton = tuple(c["name"] for c in row["label"]["tool_calls"])
        shingles = {latest[i:i + 5] for i in range(max(0, len(latest) - 4))}
        by_skeleton[skeleton].append((sample_id, shingles))
    near_pairs = 0
    for candidates in by_skeleton.values():
        for index, (a, x) in enumerate(candidates):
            if len(x) < 16:
                continue
            for b, y in candidates[:index]:
                if min(len(x), len(y)) < 0.90 * max(len(x), len(y)):
                    continue
                if len(x & y) / max(1, len(x | y)) >= 0.90:
                    union(a, b)
                    near_pairs += 1
    groups = defaultdict(list)
    for sample_id in sorted(by_id):
        groups[find(sample_id)].append(sample_id)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    target = round(len(rows) * 0.20)
    validation, train = [], []
    for group in ordered:
        if len(validation) < target:
            validation.extend(group)
        else:
            train.extend(group)
    train.sort()
    validation.sort()
    train_set = set(train)
    fit_rows = [r for r in rows if r["id"] in train_set]
    selected = []

    def add(predicate):
        for row in fit_rows:
            if row["id"] not in selected and predicate(row):
                selected.append(row["id"])
                return

    add(lambda r: len(r["label"]["tool_calls"]) == 0)
    add(lambda r: len(r["label"]["tool_calls"]) == 1 and not taxonomy(r)["ask"] and not r.get("history"))
    add(lambda r: len(r["label"]["tool_calls"]) == 2 and not taxonomy(r)["ask"])
    add(lambda r: len(r["label"]["tool_calls"]) >= 5 and not taxonomy(r)["ask"])
    add(lambda r: len(r["label"]["tool_calls"]) == 1 and taxonomy(r)["ask"])
    add(lambda r: len(r["label"]["tool_calls"]) >= 2 and taxonomy(r)["ask"])
    add(lambda r: len(r.get("history", [])) >= 18)
    add(lambda r: len(r.get("history", [])) == 4)
    add(lambda r: any(c["name"] == "alarm.create" for c in r["label"]["tool_calls"]))
    add(lambda r: any(c["name"] == "sms.send" for c in r["label"]["tool_calls"]))
    add(lambda r: any(c["name"].startswith("calendar.") for c in r["label"]["tool_calls"]))
    add(lambda r: any(c["name"].startswith("smart_home.") for c in r["label"]["tool_calls"]))

    def distribution(ids):
        subset = [by_id[i] for i in ids]
        return {"samples": len(ids), "call_counts": dict(Counter(len(r["label"]["tool_calls"]) for r in subset)),
                "ask": sum(taxonomy(r)["ask"] for r in subset),
                "with_history": sum(bool(r.get("history")) for r in subset)}

    result = {"version": "split-v1", "seed": seed, "train_jsonl_sha256": source_hash,
              "grouping": "NFC/lower/whitespace exact latest and full user dialogue, exact current-audio SHA256, and character-5gram Jaccard >=0.90 within gold tool skeleton",
              "limitations": ["Not speaker-disjoint; no speaker IDs", "Near-duplicate grouping is a heuristic, not proof of no leakage", "Not stratified; slice distributions reported", "Generic shared history does not merge all conversations"],
              "train_fit": train, "val_main": validation, "smoke_train_fit": selected,
              "group_ids": {i: find(i) for i in by_id}, "near_pairs_grouped": near_pairs,
              "largest_group": max(map(len, groups.values())),
              "distributions": {"train_fit": distribution(train), "val_main": distribution(validation), "smoke": distribution(selected)}}
    assert not set(train) & set(validation)
    assert not {find(i) for i in train} & {find(i) for i in validation}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({"manifest": str(MANIFEST), "distributions": result["distributions"],
                      "largest_group": result["largest_group"], "near_pairs": result["near_pairs_grouped"]}, indent=2))
