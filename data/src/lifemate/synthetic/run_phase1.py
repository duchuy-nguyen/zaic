import argparse
import json

from .builder import SyntheticBuilder
from .common import OFFICIAL, RUNS, load_catalog, load_fit_ids, load_jsonl

DEFAULT_TARGET = {"t1": 500, "t2": 150, "t3": 375, "t4c": 400}


def load_conf():
    p = RUNS / "asr_cache" / "confusion_fit_v1.json"
    if not p.exists():
        raise SystemExit("Chưa có confusion_fit_v1.json -> chạy: python -m src.lifemate.synthetic.t4b_confusion")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True, choices=["t1", "t2", "t3", "t4c"])
    ap.add_argument("--target", type=int, default=0, help="0 = mặc định theo tech")
    ap.add_argument("--mode", choices=["rule"], default="rule", help="t3: chỉ còn rule (llm đã bỏ)")
    a = ap.parse_args()
    target = a.target or DEFAULT_TARGET[a.tech]
    fit = load_fit_ids()
    samples = [s for s in load_jsonl(OFFICIAL / "train" / "train.jsonl") if s["id"] in fit]
    cat = load_catalog("train")
    b = SyntheticBuilder(a.tech, fit)
    if a.tech == "t1":
        from .t1_zero_action import build_t1
        n = build_t1(b, samples, cat, target)
    elif a.tech == "t2":
        from .t2_state_mutation import build_t2
        n = build_t2(b, samples, cat, target)
    elif a.tech == "t3":
        from .t3_param_removal import build_t3
        n = build_t3(b, samples, cat, target)
    else:
        from .t4_asr_noise import build_t4
        n = build_t4(b, samples, cat, load_conf(), target)
    print(b.finalize(), "accepted:", n)


if __name__ == "__main__":
    main()
