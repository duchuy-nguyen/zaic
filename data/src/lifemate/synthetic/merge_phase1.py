"""Gộp các phase1_*.jsonl thành synthetic_phase1.jsonl, CÓ GIỚI HẠN TỈ LỆ khi trộn với TRAIN_FIT.

Bản cũ gộp hết: 205 rỗng + 280 ask + 3 có hành động -> trộn vào train thì tỉ lệ rỗng 7.2% -> 15.5%,
ask 14% -> 24.4%, trong khi gold rỗng/ask sai đều bị 0 điểm.
Bản mới: chọn số mẫu t1 (rỗng) và t2+t3 (ask) sao cho sau khi trộn
    tỉ lệ rỗng <= CAP x tỉ lệ rỗng của FIT,   tỉ lệ ask <= CAP x tỉ lệ ask của FIT   (CAP mặc định 1.3)
và giữ tỉ lệ t2 (thực thể mơ hồ) : t3 (thiếu giá trị) gần tỉ lệ của train. t4 (có hành động) lấy hết.
Các file phase1_*.jsonl vẫn giữ nguyên làm pool; chọn lại được bằng --cap.
"""
import argparse
import json
import random
from collections import Counter

from .common import MANIFEST, OFFICIAL, SYN_ROOT, load_catalog, load_fit_ids, load_jsonl, tool_map

FILES = {"t1_zero_action": "t1", "t2_state_mutation": "t2", "t3_param_removal": "t3", "t4_asr_noise": "t4c"}


def behavior(calls):
    if not calls:
        return "empty"
    return "ask" if any(c["name"] == "sys.ask_missing_info" for c in calls) else "action"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool = {t: (load_jsonl(SYN_ROOT / f"phase1_{f}.jsonl") if (SYN_ROOT / f"phase1_{f}.jsonl").exists() else [])
            for t, f in FILES.items()}
    for v in pool.values():
        rng.shuffle(v)
    fit = load_fit_ids()
    train = [s for s in load_jsonl(OFFICIAL / "train" / "train.jsonl") if s["id"] in fit]
    tmap = tool_map(load_catalog("train"))
    N = len(train)
    b = Counter(behavior(s["label"]["tool_calls"]) for s in train)
    E, A = b["empty"], b["ask"]
    ent = sca = 0
    for s in train:
        c = s["label"]["tool_calls"]
        if c and c[-1]["name"] == "sys.ask_missing_info":
            props = tmap[c[-1]["arguments"]["tool_name"]]["parameters"].get("properties", {})
            if any("x-ref" in props.get(p, {}) for p in c[-1]["arguments"]["params"]):
                ent += 1
            else:
                sca += 1
    e_cap, a_cap = a.cap * E / N, a.cap * A / N
    z = len(pool["t4_asr_noise"])
    x, y = len(pool["t1_zero_action"]), len(pool["t2_state_mutation"]) + len(pool["t3_param_removal"])
    while True:
        T = N + x + y + z
        ok_e, ok_a = (E + x) / T <= e_cap + 1e-9, (A + y) / T <= a_cap + 1e-9
        if ok_e and ok_a:
            break
        if not ok_e and x > 0:
            x -= 1
        if not ok_a and y > 0:
            y -= 1
        if x == 0 and y == 0:
            break
    # chia y cho t2 : t3 theo tỉ lệ train (thực thể mơ hồ : thiếu giá trị)
    n2_pool, n3_pool = len(pool["t2_state_mutation"]), len(pool["t3_param_removal"])
    n2 = min(n2_pool, round(y * ent / max(ent + sca, 1)))
    n3 = min(n3_pool, y - n2)
    n2 = min(n2_pool, y - n3)
    pick = {"t1_zero_action": x, "t2_state_mutation": n2, "t3_param_removal": n3, "t4_asr_noise": z}
    recs = sorted((r for t, k in pick.items() for r in pool[t][:k]), key=lambda r: r["id"])

    (SYN_ROOT / "synthetic_phase1.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                                                    encoding="utf-8", newline="\n")
    T = N + len(recs)
    mix = Counter(behavior(s["label"]["tool_calls"]) for s in train) + Counter(behavior(r["label"]["tool_calls"]) for r in recs)
    meta = {"n": len(recs), "by_tech": dict(Counter(r["technique"] for r in recs)),
            "pool": {t: len(v) for t, v in pool.items()}, "cap": a.cap,
            "fit_pct": {k: round(v / N * 100, 1) for k, v in sorted(b.items())},
            "fit_plus_synthetic_pct": {k: round(v / T * 100, 1) for k, v in sorted(mix.items())},
            "train_ask_entity_vs_scalar": [ent, sca]}
    (SYN_ROOT / "synthetic_phase1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    (MANIFEST / "synth_phase1.json").write_text(json.dumps(
        [{"id": r["id"], "technique": r["technique"], "source_id": r["source_id"]} for r in recs], ensure_ascii=False),
        encoding="utf-8", newline="\n")
    print("merged:", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
