# DATA v2.1 — 551 text-training candidates

Bản nạp text-training từ gói DATA v2.1 552 mẫu, tạm tách
`synth_v21_t2_00016` để DATA phân xử cụm “Mai Anh nhà chị” với hai contact có
relation em gái / bạn cùng lớp. **Chưa kết luận mẫu này chắc chắn sai nhãn**;
đây là lựa chọn quarantine bảo thủ theo yêu cầu giữ phần có thể train trước.
Dữ liệu nguồn và nhãn các record còn lại không bị sửa.

| Nhóm | Nguồn v2.1 | Còn lại |
|---|---:|---:|
| T1 | 78 | 78 |
| T2 | 23 | 22 |
| T3 | 84 | 84 |
| T4 | 367 | 367 |
| Tổng | 552 | 551 |

## Nạp dữ liệu

File: `data/releases/synthetic_phase1_v21_text_551/synthetic_phase1.jsonl`.
Nạp như phần **synthetic bổ sung cho TRAIN_FIT**, không thay toàn bộ train gốc.
Pin manifest split DATA v2.1 đúng SHA256 trong release.json: FIT=1237, VAL=308.
Nếu dùng toàn bộ FIT này, tổng là 1237+551=1788 record text. Không cộng chồng
cả bộ synthetic 311 cũ và bộ này: các nguồn/mẫu có overlap.

```sh
python3.11 data/releases/synthetic_phase1_v21_text_551/verify_release.py
```

Giữ nguồn và các biến thể cùng source ở cùng partition, không chia lại synthetic
thành FIT/VAL độc lập để đánh giá. Chưa có source train gốc trong gói để tự xác
minh leakage/provenance; hash manifest không thay kiểm tra nội dung nguồn.

## Text và audio

`training_usage.csv` có đủ 551 ID và policy từng record:

- **451 T3/T4:** audio_text đã sửa, WAV vẫn gốc; chỉ dùng text, loại khỏi gold
  audio–transcript training pairs.
- **100 T1/T2:** text giữ nguyên nguồn; audio chưa được cung cấp/kiểm riêng,
  không tự coi là ASR-ready.

Release Git không bundle WAV/state hoặc weights. Người chạy replay cần state
v2.1 và đúng runtime/catalog do DATA bàn giao; text-training loader không cần
execute ToolEnv hoặc đọc state. Không đưa label/state vào inference model input.

## Kiểm chứng

- Hash nguồn 552 record được khóa; kết quả 551 record là phép bỏ đúng một ID,
  giữ nguyên nội dung và thứ tự record.
- Static checker hiện tại: 0 error / 0 review / 0 advisory.
- Replay mới: 473/473 mẫu có call khớp results, changed final_state và call log;
  78 zero-call labels hợp lệ.
- Verifier kiểm checksum, ID uniqueness, counts, exclusion và training usage;
  có thể tái tạo từ JSONL nguồn với --source/--output, từ chối ghi đè.
- validation.json ghi phạm vi và các kiểm tra chưa làm. 167 dòng audit của DATA
  nguồn có verdict trống; không coi đó là adjudication đã hoàn tất.

**Đây là text-training candidate đã kiểm kỹ thuật, chưa chứng nhận toàn bộ nhãn
đúng hoặc train sẽ tốt hơn.** Chưa có independent semantic adjudication, kiểm
train-state provenance/VAL text leakage, audio realism hoặc kết quả held-out VAL.
Không áp metric fixture của checker thành semantic accuracy của DATA này.

## Tái tạo từ nguồn

```sh
python3.11 data/releases/synthetic_phase1_v21_text_551/verify_release.py \
  --source /path/to/v21/synthetic_phase1.jsonl \
  --output /path/to/new/filtered.jsonl
```

Nguồn đúng SHA256 được lưu trong release.json, cùng tên/checksum ZIP bàn giao.
Một nguồn khác phải được review lại; verifier không tự áp exclusion sang release mới.
