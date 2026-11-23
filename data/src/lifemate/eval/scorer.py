"""DATA evaluator — cài đặt công thức chấm của đề (ảnh "Metrics"/"Formula"), dùng cho TRAIN/VAL nội bộ.

OPS CHECKLIST: "Evaluator do DATA phụ trách"; "Evaluator/scorer version và references train-only tách inference".

Công thức (đề):
  * Gold rỗng: điểm 1 khi và chỉ khi danh sách call nộp cũng rỗng (không xét hash).
  * Ngược lại: 0 nếu state_hash nộp != state_hash gold hoặc M = 0; còn lại 2PR/(P+R),
    P = M/(M+U), R = M/|G|; U = call nộp không khớp (kể cả call bị từ chối do sai schema).
  * Call khớp: cùng tên; mọi tham số KHÔNG fuzzy giống hệt; mọi tham số fuzzy (catalog x-match: fuzzy)
    có token-F1 > 0.6 sau lowercase + Unicode NFC. Mỗi call khớp tối đa một call; không xét thứ tự.
  * sys.ask_missing_info chấm như mọi tool.  Final = 100/N * sum(score).

Điểm đề KHÔNG nói rõ (ghi thành giả định, chọn được bằng tham số):
  A1 tokenizer fuzzy: "whitespace" (mặc định: NFC+lower rồi tách khoảng trắng) hoặc "word"
     (regex [^\\W_]+, bỏ dấu câu — giống llm/local_baseline/scoring.py local-spec-v1).
  A2 tập tham số phải trùng nhau (thiếu/thừa tham số = không khớp); chỉ tham số string cấp 1 mới fuzzy.
  A3 ghép cặp: maximum bipartite matching (tối đa M), giữ call trùng lặp.
  A4 mẫu chạy lỗi/timeout/thiếu trong submission: 0 điểm.

  python -m src.lifemate.eval.scorer --submission sub.jsonl --references refs.jsonl --catalog train_tools_catalog.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict

VERSION = "data-eval-v1"
ASSUMPTIONS = {
    "A1_tokenizer": "whitespace (default) | word",
    "A2_argument_keys": "exact key sets; fuzzy only for top-level string params with x-match=fuzzy",
    "A3_matching": "maximum bipartite one-to-one",
    "A4_missing_or_failed": "score 0",
}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def tokens(s: str, mode: str = "whitespace") -> Counter:
    s = _norm(s)
    return Counter(s.split() if mode == "whitespace" else re.findall(r"[^\W_]+", s))


def token_f1(a: str, b: str, mode: str = "whitespace") -> float:
    x, y = tokens(a, mode), tokens(b, mode)
    if not x or not y:
        return float(x == y)
    inter = sum((x & y).values())
    if not inter:
        return 0.0
    p, r = inter / sum(x.values()), inter / sum(y.values())
    return 2 * p * r / (p + r)


def _identical(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    return a == b


def fuzzy_params(catalog: dict) -> dict:
    return {t["name"]: {k for k, v in t.get("parameters", {}).get("properties", {}).items()
                        if v.get("x-match") == "fuzzy"}
            for a in catalog["apps"] for t in a.get("tools", [])}


def call_match(pred: dict, gold: dict, fz: dict, mode: str) -> bool:
    if not isinstance(pred, dict) or pred.get("name") != gold.get("name"):
        return False
    pa, ga = pred.get("arguments"), gold.get("arguments", {})
    if not isinstance(pa, dict) or set(pa) != set(ga):
        return False
    fset = fz.get(gold["name"], set())
    for k, gv in ga.items():
        pv = pa[k]
        if k in fset and isinstance(gv, str) and isinstance(pv, str):
            if token_f1(pv, gv, mode) <= 0.6:
                return False
        elif not _identical(pv, gv):
            return False
    return True


def max_matches(pred: list, gold: list, fz: dict, mode: str) -> int:
    edges = [[j for j, g in enumerate(gold) if call_match(p, g, fz, mode)] for p in pred]
    owner: dict[int, int] = {}

    def augment(i, seen):
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(pred)))


def score_sample(pred_calls, pred_hash, gold_calls, gold_hash, fz, mode="whitespace", failed=False) -> dict:
    if failed or pred_calls is None:
        return {"score": 0.0, "M": 0, "U": 0, "G": len(gold_calls), "state_match": False, "reason": "missing_or_failed"}
    if not gold_calls:
        ok = not pred_calls
        return {"score": float(ok), "M": 0, "U": len(pred_calls), "G": 0, "state_match": None,
                "reason": "empty_gold_ok" if ok else "empty_gold_but_calls"}
    M = max_matches(pred_calls, gold_calls, fz, mode)
    U = len(pred_calls) - M
    state = pred_hash == gold_hash
    if not state or M == 0:
        return {"score": 0.0, "M": M, "U": U, "G": len(gold_calls), "state_match": state,
                "reason": "state_mismatch" if not state else "no_match"}
    P, R = M / (M + U), M / len(gold_calls)
    return {"score": 2 * P * R / (P + R), "M": M, "U": U, "G": len(gold_calls), "state_match": True, "reason": "ok"}


def behavior(calls) -> str:
    if not calls:
        return "empty"
    return "ask" if any(c.get("name") == "sys.ask_missing_info" for c in calls) else "action"


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate(submission: list, references: list, catalog: dict, mode="whitespace") -> dict:
    fz = fuzzy_params(catalog)
    refs = {r["id"]: r for r in references}
    sub = {}
    dup = [r["id"] for r in submission if r["id"] in sub or sub.setdefault(r["id"], r) is not r]
    extra = sorted(set(sub) - set(refs))
    per, by = [], defaultdict(list)
    for sid, ref in refs.items():
        s = sub.get(sid)
        res = score_sample(None if s is None else s.get("tool_calls", []), None if s is None else s.get("state_hash"),
                           ref["tool_calls"], ref["state_hash"], fz, mode, failed=bool(s and s.get("failed")))
        res["id"] = sid
        per.append(res)
        by[behavior(ref["tool_calls"])].append(res["score"])
    n = len(refs)
    return {"scorer_version": VERSION, "tokenizer": mode, "assumptions": ASSUMPTIONS,
            "n_references": n, "final_score": round(100 * sum(r["score"] for r in per) / n, 4) if n else None,
            "by_gold_behavior": {k: {"n": len(v), "score": round(100 * sum(v) / len(v), 4)} for k, v in sorted(by.items())},
            "missing_ids": sum(1 for r in per if r["reason"] == "missing_or_failed"),
            "extra_ids": extra, "duplicate_ids": dup,
            "reasons": dict(Counter(r["reason"] for r in per)), "per_sample": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--references", required=True, help="train-only: id, tool_calls, state_hash (không đưa vào inference)")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--tokenizer", choices=["whitespace", "word"], default="whitespace")
    ap.add_argument("--per-sample", default=None, help="ghi điểm từng mẫu ra JSONL")
    a = ap.parse_args()
    for s in (sys.stdout,):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rep = evaluate(load_jsonl(a.submission), load_jsonl(a.references),
                   json.load(open(a.catalog, encoding="utf-8")), a.tokenizer)
    if a.per_sample:
        with open(a.per_sample, "w", encoding="utf-8", newline="\n") as f:
            for r in rep["per_sample"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep.pop("per_sample")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    sys.exit(2 if rep["extra_ids"] or rep["duplicate_ids"] else 0)


if __name__ == "__main__":
    main()
