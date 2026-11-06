"""Bảng nhầm lẫn ASR gold-token -> predicted-token, CHỈ từ audio của TRAIN_FIT hiện tại.

Sửa so với bản cũ:
  * Dựng lại theo manifest mới (bản cũ dựng trên FIT cũ, nay một phần là VAL -> rò thông tin VAL vào t4).
  * Token chuẩn hoá bỏ dấu câu ('tiếng,' -> 'tiếng'); bỏ cặp gold == predicted.
"""
import difflib
import json
from collections import Counter, defaultdict

from .common import OFFICIAL, RUNS, load_fit_ids, load_jsonl, tokens


def main():
    fit = load_fit_ids()
    samples = [s for s in load_jsonl(OFFICIAL / "train" / "train.jsonl") if s["id"] in fit]
    gold = {}
    for s in samples:
        gold[s["audio"]] = s["audio_text"]
        for t in s.get("history", []):
            if t.get("role") == "user" and t.get("audio"):
                gold[t["audio"]] = t.get("audio_text", "")
    pred = {}
    for l in (RUNS / "asr_cache" / "transcripts.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        pred[r["audio"]] = r["text"]
    conf = defaultdict(Counter)
    used = 0
    for a, g in gold.items():
        if a not in pred:
            continue
        used += 1
        gt, pt = tokens(g), tokens(pred[a])
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gt, pt, autojunk=False).get_opcodes():
            if tag == "replace" and (i2 - i1) == (j2 - j1):
                for x, y in zip(gt[i1:i2], pt[j1:j2]):
                    if x != y:
                        conf[x][y] += 1
    out = {a: dict(c.most_common(5)) for a, c in conf.items() if sum(c.values()) >= 2}
    (RUNS / "asr_cache" / "confusion_fit_v1.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"audio FIT có transcript: {used}/{len(gold)} | confusion entries: {len(out)} -> confusion_fit_v1.json")


if __name__ == "__main__":
    main()
