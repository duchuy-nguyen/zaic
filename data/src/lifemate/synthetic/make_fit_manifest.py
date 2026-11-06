"""Chia TRAIN thành TRAIN_FIT / VAL_MAIN (split-v1), theo mục 09 của plan 03/11.

Sửa so với bản cũ (group theo SHA256 của `history`):
  * Bản cũ gộp cả 527 mẫu KHÔNG có history vào MỘT nhóm (cùng hash của `[]`)
    -> VAL không có mẫu nào không có history, VAL chỉ còn 13% thay vì ~20%.
  * Bản mới: group theo HỘI THOẠI (union-find), mẫu không có history là nhóm riêng,
    rồi chia group-aware + phân tầng (hành vi x độ dài history x số call).

Quy tắc group (nối 2 mẫu vào cùng group khi):
  R1. Trùng transcript hiện tại sau NFC/lowercase/gộp khoảng trắng (vd train_0061/0066).
  R2. Trùng toàn bộ hội thoại (history + câu hiện tại).
  R3. Chuỗi câu user của mẫu A (history + hiện tại) là TIỀN TỐ chuỗi câu user trong
      history của mẫu B (A là một bước trước của cùng hội thoại), độ dài tiền tố >= 2.
  Không group theo một câu chung chung đơn lẻ trùng nhau trong history (plan mục 09.4),
  chỉ báo cáo overlap.

Chạy:
  python -m src.lifemate.synthetic.make_fit_manifest            # dùng OFFICIAL trong common.py
  python -m src.lifemate.synthetic.make_fit_manifest --train <train.jsonl> --out <dir>
Đầu ra:
  manifests/split_v1.json                 # đầy đủ: ids, groups, strata, seed, sha256, rules
  manifests/train_fit_provisional.json    # danh sách id FIT (giữ tên cũ để code khác không phải sửa)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261103
VAL_RATIO = 0.20
VERSION = "split-v1"


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "").lower()).strip()


def user_turns(r: dict) -> list[str]:
    return [norm(h.get("audio_text")) for h in r.get("history", []) if h.get("role") == "user"]


def behavior(r: dict) -> str:
    calls = r["label"]["tool_calls"]
    if not calls:
        return "empty"
    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        return "ask"
    return "action"


def hist_bin(r: dict) -> str:
    n = len(r.get("history", []))
    return "h0" if n == 0 else ("h1-4" if n <= 4 else "h5+")


def calls_bin(r: dict) -> str:
    n = len([c for c in r["label"]["tool_calls"] if not c["name"].startswith("sys.")])
    return "c0" if n == 0 else ("c1" if n == 1 else ("c2" if n == 2 else "c3+"))


def domain(r: dict) -> str:
    for c in r["label"]["tool_calls"]:
        if not c["name"].startswith("sys."):
            return c["name"].split(".")[0]
    return "-"


class DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # gốc = id nhỏ nhất, để group id ổn định
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def build_groups(rows: list[dict]) -> dict[str, list[str]]:
    ids = [r["id"] for r in rows]
    dsu = DSU(ids)
    by_cur, by_dialog = defaultdict(list), defaultdict(list)
    seq = {}
    for r in rows:
        by_cur[norm(r["audio_text"])].append(r["id"])
        dialog = json.dumps([user_turns(r), norm(r["audio_text"])], ensure_ascii=False)
        by_dialog[dialog].append(r["id"])
        seq[r["id"]] = user_turns(r) + [norm(r["audio_text"])]
    for bucket in (by_cur, by_dialog):                       # R1, R2
        for same in bucket.values():
            for x in same[1:]:
                dsu.union(same[0], x)
    # R3: A (history+current) là tiền tố của history B
    index = defaultdict(list)                                 # prefix-key -> ids có history bắt đầu như vậy
    for r in rows:
        h = user_turns(r)
        for k in range(2, len(h) + 1):
            index[json.dumps(h[:k], ensure_ascii=False)].append(r["id"])
    for r in rows:
        s = seq[r["id"]]
        if len(s) >= 2:
            for other in index.get(json.dumps(s, ensure_ascii=False), []):
                if other != r["id"]:
                    dsu.union(r["id"], other)
    groups = defaultdict(list)
    for i in ids:
        groups[dsu.find(i)].append(i)
    return {g: sorted(v) for g, v in groups.items()}


def split(rows: list[dict], seed=SEED, ratio=VAL_RATIO):
    byid = {r["id"]: r for r in rows}
    groups = build_groups(rows)
    stratum_of = {}
    strata = defaultdict(list)                                # stratum -> [group ids]
    for g, members in groups.items():
        r0 = byid[members[0]]
        key = f"{behavior(r0)}|{hist_bin(r0)}|{calls_bin(r0)}"
        stratum_of[g] = key
        strata[key].append(g)
    rng = random.Random(seed)
    val_ids = set()
    for key in sorted(strata):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        n_samples = sum(len(groups[g]) for g in gs)
        target = round(n_samples * ratio)
        taken = 0
        for g in gs:
            if taken >= target:
                break
            if taken + len(groups[g]) > target + 1:           # không vượt quá 1 mẫu
                continue
            val_ids.update(groups[g])
            taken += len(groups[g])
    fit_ids = sorted(i for i in byid if i not in val_ids)
    return fit_ids, sorted(val_ids), groups, stratum_of


def dist(rows, fn):
    c = Counter(fn(r) for r in rows)
    n = len(rows) or 1
    return {k: round(v / n * 100, 1) for k, v in sorted(c.items())}


def shingles(s, k=5):
    s = norm(s)
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def overlap_report(fit, val):
    """Chỉ BÁO CÁO (không group): near-duplicate transcript và câu history dùng chung giữa FIT/VAL."""
    fs = [(r["id"], shingles(r["audio_text"])) for r in fit]
    near = 0
    for v in val:
        sv = shingles(v["audio_text"])
        for _, sf in fs:
            j = len(sv & sf) / (len(sv | sf) or 1)
            if j >= 0.9:
                near += 1
                break
    fh = Counter(t for r in fit for t in user_turns(r))
    shared_hist = sum(1 for v in val for t in user_turns(v) if t in fh)
    return {"val_near_dup_jaccard>=0.9": near, "val_history_turns_also_in_fit": shared_hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None, help="đường dẫn train.jsonl (mặc định OFFICIAL/train/train.jsonl)")
    ap.add_argument("--out", default=None, help="thư mục manifests (mặc định MANIFEST trong common.py)")
    a = ap.parse_args()
    if a.train is None or a.out is None:
        from .common import OFFICIAL, MANIFEST
        train_path = Path(a.train) if a.train else OFFICIAL / "train" / "train.jsonl"
        out = Path(a.out) if a.out else MANIFEST
    else:
        train_path, out = Path(a.train), Path(a.out)

    raw = train_path.read_bytes()
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    fit_ids, val_ids, groups, stratum_of = split(rows)
    byid = {r["id"]: r for r in rows}
    fit = [byid[i] for i in fit_ids]
    val = [byid[i] for i in val_ids]

    # kiểm tra không group nào bị tách đôi
    side = {i: "fit" for i in fit_ids} | {i: "val" for i in val_ids}
    broken = [g for g, m in groups.items() if len({side[i] for i in m}) > 1]
    assert not broken, f"group bị tách: {broken[:5]}"

    report = {
        "n": {"total": len(rows), "fit": len(fit), "val": len(val),
              "val_pct": round(len(val) / len(rows) * 100, 1),
              "groups": len(groups), "multi_sample_groups": sum(len(m) > 1 for m in groups.values()),
              "largest_group": max(len(m) for m in groups.values())},
        "behavior_pct": {"fit": dist(fit, behavior), "val": dist(val, behavior)},
        "history_pct": {"fit": dist(fit, hist_bin), "val": dist(val, hist_bin)},
        "calls_pct": {"fit": dist(fit, calls_bin), "val": dist(val, calls_bin)},
        "domains_only_in_val": sorted({domain(r) for r in val} - {domain(r) for r in fit}),
        "domains_only_in_fit": sorted({domain(r) for r in fit} - {domain(r) for r in val}),
        "overlap_report_only": overlap_report(fit, val),
    }
    manifest = {
        "version": VERSION, "seed": SEED, "val_ratio": VAL_RATIO,
        "source": {"file": train_path.name, "sha256": hashlib.sha256(raw).hexdigest()},
        "rules": ["R1 dup current transcript (NFC/lower/space)", "R2 dup full dialogue",
                  "R3 user-turn prefix >= 2 (same conversation, earlier step)",
                  "strata = behavior x history_bin x calls_bin; group-aware; seeded shuffle"],
        "fit": fit_ids, "val": val_ids,
        "groups": {g: m for g, m in sorted(groups.items()) if len(m) > 1},
        "report": report,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    (out / "train_fit_provisional.json").write_text(json.dumps(fit_ids), encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"-> {out/'split_v1.json'} | {out/'train_fit_provisional.json'}  (FIT {len(fit)} / VAL {len(val)})")


if __name__ == "__main__":
    main()
