"""Đóng gói metadata release (KHÔNG chứa data) cho synthetic phase1 v2.1.

Ghi synthetic/release_v21.json: hash dataset/manifest/confusion, số lượng theo kỹ thuật, pool, tỉ lệ trộn,
kết quả validate, chính sách. File này (và README) để trên git; data (.jsonl, synthetic_environment) để trên Drive.

  python -m src.lifemate.synthetic.validate_phase1 --file synthetic/synthetic_phase1.jsonl --json-out synthetic/validation_v21.json
  python -m src.lifemate.synthetic.release_phase1
"""
import hashlib
import json
from pathlib import Path

from .common import MANIFEST, RUNS, SYN_ROOT, load_jsonl


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main():
    data = SYN_ROOT / "synthetic_phase1.jsonl"
    meta = json.loads((SYN_ROOT / "synthetic_phase1_meta.json").read_text(encoding="utf-8"))
    val_p = SYN_ROOT / "validation_v21.json"
    val = json.loads(val_p.read_text(encoding="utf-8")) if val_p.exists() else None
    if not val or val.get("sha256") != sha(data):
        raise SystemExit("Chạy validate_phase1 --file synthetic/synthetic_phase1.jsonl --json-out synthetic/validation_v21.json trước.")
    if val["error_count"]:
        raise SystemExit(f"validate còn {val['error_count']} lỗi -> không release.")
    recs = load_jsonl(data)
    rel = {
        "release_version": "synthetic_phase1_v21",
        "supersedes": ["synthetic_phase1_v2 (331, sha256 b27d7e38...)", "synthetic_phase1_v2_filtered_311 (OPS)"],
        "id_prefix": "synth_v21_",
        "records": len(recs),
        "by_technique": meta["by_tech"],
        "pool": meta["pool"],
        "mix_cap": meta["cap"],
        "fit_pct": meta["fit_pct"],
        "fit_plus_synthetic_pct": meta["fit_plus_synthetic_pct"],
        "dataset_sha256": sha(data),
        "split_manifest_sha256": sha(MANIFEST / "split_v1.json"),
        "train_fit_ids_sha256": sha(MANIFEST / "train_fit_provisional.json"),
        "confusion_sha256": sha(RUNS / "asr_cache" / "confusion_fit_v1.json"),
        "validation": {"error_count": val["error_count"], "checks": val["checks"], "replay": val["replay"],
                       "not_checked": val["not_checked"]},
        "text_only_records": val["text_only_records"],
        "text_only_rule": "t3_param_removal và t4_asr_noise: audio_text khác WAV -> chỉ train text, không train ASR/audio",
        "policy": [
            "t2: chỉ nhân đôi TÊN RIÊNG trơn (như train: Minh/Lan/Hương); bản sao khác id + phone + relation",
            "t2: bỏ mẫu có từ chỉ vai trò quanh tên trùng/khớp relation (OPS: synth_t2_00014, synth_t2_00047)",
            "t2: không sinh bản sao chỉ-khác-ID (OPS loại 18 mẫu kiểu này)",
            "t4: tối đa 2 biến thể / câu nguồn, mỗi biến thể đổi một token gốc khác nhau",
        ],
        "data_location": "Drive ZAIC/data/synthetic_phase1_v21 (không để data trên git)",
        "full_semantic_quality_certified": False,
        "semantic_audit": "synthetic/audit_phase1.csv (toàn bộ t2, t3 + mẫu t4, t1) — cần người duyệt",
    }
    out = SYN_ROOT / "release_v21.json"
    out.write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    names = {"t1_zero_action": "t1 không có app → []", "t2_state_mutation": "t2 trùng tên → hỏi lại",
             "t3_param_removal": "t3 thiếu số tiền/giờ → hỏi lại", "t4_asr_noise": "t4 nhiễu ASR tên thực thể"}
    rows = "\n".join(f"| {names[t]} | {meta['pool'].get(t, 0)} | {meta['by_tech'].get(t, 0)} |" for t in names)
    fp, mp = meta["fit_pct"], meta["fit_plus_synthetic_pct"]
    md = f"""# Release synthetic_phase1_v21 (tự sinh bởi release_phase1.py — không sửa tay)

| Kỹ thuật | Pool | Trong bản gộp |
|---|---:|---:|
{rows}
| **Tổng** | **{sum(meta['pool'].values())}** | **{len(recs)}** |

| | Rỗng | Hỏi lại | Hành động |
|---|---:|---:|---:|
| TRAIN_FIT | {fp.get('empty')}% | {fp.get('ask')}% | {fp.get('action')}% |
| TRAIN_FIT + synthetic v2.1 | {mp.get('empty')}% | {mp.get('ask')}% | {mp.get('action')}% |

- Validate: **{val['error_count']} lỗi**, {len(val['checks'])} nhóm kiểm tra, replay {val['replay']}.
- Mẫu chỉ train text (t3, t4): {val['text_only_records']}.
- dataset sha256: `{rel['dataset_sha256']}`
- split manifest sha256: `{rel['split_manifest_sha256']}`
- confusion sha256: `{rel['confusion_sha256']}`
"""
    (SYN_ROOT / "release_v21.md").write_text(md, encoding="utf-8", newline="\n")
    print(json.dumps({k: rel[k] for k in ("records", "by_technique", "fit_plus_synthetic_pct", "dataset_sha256")},
                     ensure_ascii=False))
    print("->", out)


if __name__ == "__main__":
    main()
