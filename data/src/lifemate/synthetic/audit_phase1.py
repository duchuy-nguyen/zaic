"""Xuất bảng kiểm duyệt NGỮ NGHĨA cho người (OPS: "full_independent_semantic_adjudication" chưa có).

Máy chỉ chứng minh được nhãn chạy đúng (validate_phase1). Câu hỏi "nhãn có đúng ý người nói không" cần người đọc.
Bảng gồm: TOÀN BỘ t2 và t3 (rủi ro cao nhất: nhãn hỏi lại), 40 mẫu t4 và 20 mẫu t1 chọn ngẫu nhiên có seed.
Mỗi dòng có câu hỏi kiểm tra cụ thể; người duyệt điền cột verdict = OK / SAI / NGHI và ghi chú.
Mở bằng Excel (UTF-8 có BOM).

  python -m src.lifemate.synthetic.audit_phase1 --file synthetic/synthetic_phase1.jsonl --out synthetic/audit_phase1.csv
  python -m src.lifemate.synthetic.audit_phase1 --apply synthetic/audit_phase1.csv     # sau khi duyệt: in thống kê + id SAI
"""
import argparse
import csv
import json
import random
from collections import Counter

from .common import load_jsonl

QUESTION = {
    "t1_zero_action": "Không còn app nào làm được yêu cầu này với danh sách app hiện có? (kể cả app khác tên)",
    "t2_state_mutation": "Câu nói có chi tiết nào chỉ ra đúng MỘT trong hai người/thực thể trùng tên không? Nếu có -> SAI.",
    "t3_param_removal": "Câu mới còn nói (trực tiếp/gián tiếp) giá trị đã lược không? Câu có tự nhiên không?",
    "t4_asr_noise": "Cụm tên bị nhiễu có còn nhận ra đúng thực thể không? Nhiễu có giống lỗi nghe nhầm thật không?",
}


def _calls(r):
    out = []
    for c in r["label"]["tool_calls"]:
        a = json.dumps(c["arguments"], ensure_ascii=False)
        out.append(f"{c['name']}({a[:80]})")
    return " -> ".join(out) or "[] (không gọi tool)"


def _evidence(r):
    p, t = r.get("provenance", {}), r["technique"]
    if t == "t1_zero_action":
        return "gỡ app: " + ", ".join(p.get("removed_apps", []))
    if t == "t2_state_mutation":
        ch = {k: v for k, v in p.get("changed", {}).items()}
        return f"trùng tên '{p.get('duplicate_name')}' ({p.get('entity_list')}); gốc vs bản sao: {json.dumps(ch, ensure_ascii=False)}"
    if t == "t3_param_removal":
        return f"lược {p.get('target_tool')}.{p.get('removed_param')} = {p.get('removed_value')}"
    if t == "t4_asr_noise":
        op = (p.get("noise_ops") or [{}])[0]
        return f"'{op.get('gold')}' -> '{op.get('noisy')}' | '{p.get('entity_name')}' -> '{p.get('noisy_mention')}' (sim {p.get('ground_sim')})"
    return ""


def export(path, out, seed=20261107):
    recs = load_jsonl(path)
    rng = random.Random(seed)
    by = {t: [r for r in recs if r["technique"] == t] for t in QUESTION}
    pick = by["t2_state_mutation"] + by["t3_param_removal"] + \
        rng.sample(by["t4_asr_noise"], min(40, len(by["t4_asr_noise"]))) + \
        rng.sample(by["t1_zero_action"], min(20, len(by["t1_zero_action"])))
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "technique", "source_id", "text_original", "text_synthetic", "label_calls",
                    "evidence", "question", "verdict", "note"])
        for r in pick:
            orig = r.get("provenance", {}).get("original_text") or r["audio_text"]
            w.writerow([r["id"], r["technique"], r["source_id"], orig, r["audio_text"], _calls(r),
                        _evidence(r), QUESTION[r["technique"]], "", ""])
    print(f"{len(pick)} dòng -> {out} | " + ", ".join(f"{t}: {sum(1 for r in pick if r['technique'] == t)}" for t in QUESTION))


def apply(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    v = Counter((r["technique"], (r["verdict"] or "-").strip().upper()) for r in rows)
    print(dict(v))
    bad = [r["id"] for r in rows if (r["verdict"] or "").strip().upper() in ("SAI", "NGHI")]
    print("SAI/NGHI:", bad)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="synthetic/synthetic_phase1.jsonl")
    ap.add_argument("--out", default="synthetic/audit_phase1.csv")
    ap.add_argument("--apply", default=None, help="đọc CSV đã duyệt, in thống kê và id SAI/NGHI")
    a = ap.parse_args()
    if a.apply:
        apply(a.apply)
    else:
        export(a.file, a.out)


if __name__ == "__main__":
    main()
