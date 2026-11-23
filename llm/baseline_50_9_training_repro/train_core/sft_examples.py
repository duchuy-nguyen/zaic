"""TRAIN-only next-action SFT, exact frozen split; no ToolEnv or public data access."""
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 20261103


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)


def make_examples(row, catalog, system, initial_messages, allowed_schemas):
    view = {"id": row["id"], "latest_text": row["audio_text"], "context": row["context"],
            "history": [({"role": "user", "text": t["audio_text"]} if t["role"] == "user"
                         else {k:t[k] for k in ("role", "text", "tool_calls", "tool_results") if k in t})
                        for t in row.get("history", [])]}
    prompt = initial_messages(view)
    assert prompt[0]["content"] == system
    tools = allowed_schemas(view, catalog)
    available = {t["function"]["name"] for t in tools}
    calls, observations = row["label"]["tool_calls"], row["label"]["tool_results"]
    if len(calls) != len(observations):
        raise ValueError("Gold call/result count mismatch: " + row["id"])
    examples = []
    for step, (call, observation) in enumerate(zip(calls, observations)):
        if call["name"] not in available or not isinstance(call["arguments"], dict):
            raise ValueError("Invalid/unavailable gold tool: " + row["id"])
        if call["name"] == "sys.ask_missing_info" and step != len(calls)-1:
            raise ValueError("Gold continues after ask_missing_info")
        action = {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{step+1:03d}",
                  "type": "function", "function": copy.deepcopy(call)}]}
        examples.append({"id": f"{row['id']}:call:{step}", "source_id": row["id"], "step": step,
                         "kind": "call", "prompt": copy.deepcopy(prompt), "completion": [copy.deepcopy(action)],
                         "tools": copy.deepcopy(tools)})
        prompt += [action, {"role": "tool", "tool_call_id": f"call_{step+1:03d}",
                           "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))}]
    if not calls or calls[-1]["name"] != "sys.ask_missing_info":
        examples.append({"id": f"{row['id']}:finish", "source_id": row["id"], "step": len(calls),
                         "kind": "finish", "prompt": copy.deepcopy(prompt),
                         "completion": [{"role": "assistant", "content": ""}], "tools": copy.deepcopy(tools)})
    return examples


def encode(example, tokenizer):
    kwargs = {"tools": example["tools"], "tokenize": True, "return_dict": False, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(example["prompt"], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(example["prompt"]+example["completion"], add_generation_prompt=False, **kwargs)
    if full[:len(prompt)] != prompt:
        raise ValueError("Prompt is not a token prefix; refuse incorrect loss mask: " + example["id"])
    if not 0 < len(prompt) < len(full) <= 16384:
        raise ValueError(f"Empty target/context overflow {example['id']}: prompt={len(prompt)}, total={len(full)}; no truncation")
    return {"id": example["id"], "source_id": example["source_id"], "input_ids": full,
            "labels": [-100]*len(prompt)+full[len(prompt):], "length": len(full),
            "target_tokens": len(full)-len(prompt), "kind": example["kind"]}


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    life = root/"LifeMate"
    sys.path.insert(0, str(root/"llm"))
    from local_baseline.agent import SYSTEM, initial_messages, allowed_schemas
    train = life/"data/lifemate-v3.2.0/train+public_test/train/train.jsonl"
    split_path = life/"data/manifests/split-v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if sha(train) != split["train_jsonl_sha256"]:
        raise ValueError("Frozen TRAIN changed")
    rows = {r["id"]:r for r in (json.loads(l) for l in train.read_text(encoding="utf-8").splitlines() if l.strip())}
    if len(split["train_fit"]) != 1236 or len(split["val_main"]) != 309 or set(split["train_fit"]) & set(split["val_main"]):
        raise ValueError("Expected exact disjoint 1236/309 split")
    groups = split["group_ids"]
    if {groups[i] for i in split["train_fit"]} & {groups[i] for i in split["val_main"]}:
        raise ValueError("Group leakage")
    catalog_json = json.loads((train.parent/"train_tools_catalog.json").read_text(encoding="utf-8"))
    catalog = {t["name"]: t for a in catalog_json["apps"] for t in a["tools"]}
    manifest = {"version": "lifemate-sft-clean-v1", "base_model": REPO, "base_revision": REVISION,
                "seed": SEED, "train_jsonl_sha256": sha(train), "split_sha256": sha(split_path),
                "catalog_sha256": sha(train.parent/"train_tools_catalog.json"),
                "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "source": "Official labeled TRAIN only, clean current/history transcripts",
                "no_public_test": True, "no_hidden_state_inputs": True,
                "supervision": "next assistant action only; all prompt/history/tool-observation tokens masked",
                "finish_target": "empty assistant response terminated by native EOS", "partitions": {}}
    for name, partition in [("train", "train_fit"), ("val", "val_main")]:
        ids = split[partition]
        examples = [e for i in ids for e in make_examples(rows[i], catalog, SYSTEM, initial_messages, allowed_schemas)]
        if set(e["source_id"] for e in examples) != set(ids):
            raise ValueError("Missing source coverage")
        write_new(output/(name+".jsonl"), ''.join(json.dumps(e, ensure_ascii=False, separators=(",", ":"))+"\n" for e in examples))
        manifest["partitions"][name] = {"source_ids": ids, "source_count": len(ids), "examples": len(examples),
                                        "call_examples": sum(e["kind"]=="call" for e in examples),
                                        "finish_examples": sum(e["kind"]=="finish" for e in examples),
                                        "sha256": sha(output/(name+".jsonl"))}
    write_new(output/"split-v1.json", split_path.read_text(encoding="utf-8"))
    write_new(output/"DATA_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest
