"""Versioned local approximation of the LifeMate scoring specification, not an official scorer."""
from __future__ import annotations

from collections import Counter
import json
import re
import unicodedata

VERSION = "local-spec-v1"
ASSUMPTIONS = {
    "fuzzy_tokenizer": "NFC lowercase then Unicode word tokens via re.findall(r'[^\\W_]+'); punctuation discarded, accents retained; official tokenization unspecified",
    "fuzzy_threshold": "strictly > 0.6; multiset token F1",
    "arguments": "exact key sets; only top-level schema x-match:fuzzy string arguments use fuzzy comparison",
    "empty_gold": "empty prediction scores 1 regardless of hash; execution failures/timeouts always score 0",
    "matching": "maximum-cardinality bipartite one-to-one, duplicates preserved",
}


def token_f1(a: str, b: str) -> float:
    def tokens(value):
        return Counter(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", value).lower(), flags=re.UNICODE))
    x, y = tokens(a), tokens(b)
    if not x or not y:
        return float(x == y)
    overlap = sum((x & y).values())
    return 2 * overlap / (sum(x.values()) + sum(y.values()))


def exact(a, b) -> bool:
    # JSON structural equality; booleans must not compare equal to numeric 0/1.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def compatible(predicted: dict, gold: dict, tools: dict, *, fuzzy: bool = True) -> bool:
    if predicted.get("name") != gold.get("name"):
        return False
    p, g = predicted.get("arguments", {}), gold.get("arguments", {})
    if not isinstance(p, dict) or not isinstance(g, dict) or set(p) != set(g):
        return False
    properties = tools.get(gold["name"], {}).get("parameters", {}).get("properties", {})
    for key in g:
        if fuzzy and properties.get(key, {}).get("x-match") == "fuzzy" and isinstance(p[key], str) and isinstance(g[key], str):
            if token_f1(p[key], g[key]) <= 0.6:
                return False
        elif not exact(p[key], g[key]):
            return False
    return True


def maximum_matches(predicted: list, gold: list, tools: dict, *, fuzzy=True) -> int:
    edges = [[j for j, g in enumerate(gold) if compatible(p, g, tools, fuzzy=fuzzy)] for p in predicted]
    assigned = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(predicted)))


def score(prediction: dict, reference: dict, tools: dict, *, failed: bool = False) -> dict:
    predicted, gold = prediction["tool_calls"], reference["tool_calls"]
    matches = maximum_matches(predicted, gold, tools)
    f1 = 2 * matches / (len(predicted) + len(gold)) if predicted or gold else 1.0
    strict_matches = maximum_matches(predicted, gold, tools, fuzzy=False)
    strict_f1 = 2 * strict_matches / (len(predicted) + len(gold)) if predicted or gold else 1.0
    state_ok = prediction["state_hash"] == reference["state_hash"]
    if failed:
        value = 0.0
    elif not gold:
        value = float(not predicted)
    else:
        value = f1 if state_ok and matches else 0.0
    return {"score_spec_local": 100 * value, "tool_f1": f1, "strict_tool_f1": strict_f1,
            "state_match": state_ok, "matched_calls": matches, "predicted_calls": len(predicted),
            "gold_calls": len(gold), "failure_override": failed}
