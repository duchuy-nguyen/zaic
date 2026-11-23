# Audit pilot artifacts

Chạy từ repository root, không cần model/GPU hoặc dependency ngoài stdlib:

```sh
python3.11 ops/scripts/ops/audit_pilot.py /path/to/lifemate_pilot_v7_resume.tar.gz
```

Lệnh chỉ đọc tar.gz, không giải nén, không import source trong gói và không load
pickle/checkpoint. Hỗ trợ schema pilot oneclick v7: export_manifest, config,
split (fit_ids/val_ids/groups), ba agent JSONL và ba metrics JSON. Metadata mỗi
file tối đa 16 MiB; binary artifact được băm theo block. Không tải base weights.

Kiểm checksum artifact, unique ID, FIT/VAL ID và group disjoint, VAL membership,
cùng ID giữa ba chế độ, call/result alignment và metric khớp tổng hợp từ rollout.
Tổng hợp timeout, số mẫu vượt budget, p50/p95/max, repair và failed tool results.
p95 dùng nearest rank, p50 dùng median, khớp quy ước pilot.

Exit 0: các kiểm tra cấu trúc đạt; 1: phát hiện bất nhất; 2: archive/schema thiếu
hoặc không hợp lệ. Exit 0 không xác nhận model đủ chất lượng. deployment_ready luôn
false vì audit offline không thể chứng nhận GPU inference hoặc official replay.
Score/F1/state_match tổng hợp từ local_score có sẵn, không chấm lại gold độc lập.
Không xuất raw trace, model response, transcript hay exception text vào báo cáo.
Checksum đối chiếu manifest tự khai trong gói, không phải chữ ký nguồn tin cậy.

Gói ZAIC 2026-11-09: 14/14 artifact khớp; FIT 1236, VAL 309; mỗi chế độ 8 mẫu,
7 timeout và 7 vượt budget 30s. Điểm local base text 0, tuned text 12.5, audio 0.
Có thể dùng để chẩn đoán; chưa đủ bằng chứng thay baseline50. Không đưa VAL rollout
vào tập train. Pilot adapter cần base weights/revision, không thay private assets
manifest của bridge baseline50. Bản sửa AMP training smoke chưa được port vì chưa
có kiểm chứng CUDA tại workspace này.
