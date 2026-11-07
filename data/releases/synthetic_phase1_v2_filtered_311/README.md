# Synthetic phase1 v2 — filtered 311

Bản mới từ DATA v2.1: [551 text-training candidates](../synthetic_phase1_v21_text_551/RELEASE.md).
Chọn một release; không cộng cả hai vào train vì có overlap nguồn.

Bản text-training candidate từ merge DATA v2 331 mẫu: loại 2 mẫu có xung đột
qualifier/ask độ tin cậy cao và loại thêm 18 bản sao chỉ khác ID theo yêu cầu
người phụ trách phiên làm việc. **18 mẫu chỉ-ID không được coi là nhãn sai đã
xác nhận.** Dữ liệu gốc trên Drive không thay đổi.

| Nhóm | Trước | Còn lại |
|---|---:|---:|
| T1 zero-action | 57 | 57 |
| T2 ambiguity | 59 | 39 |
| T3 param-removal | 53 | 53 |
| T4 ASR-noise | 162 | 162 |
| Tổng | 331 | 311 |

## Sử dụng

Từ gốc repo:

```sh
python3.11 data/releases/synthetic_phase1_v2_filtered_311/verify_release.py
```

Nạp `data/releases/synthetic_phase1_v2_filtered_311/synthetic_phase1.jsonl` làm
phần synthetic bổ sung cho TRAIN_FIT. Không lấy 311 mẫu thay toàn bộ train gốc.
Giữ split DATA v2 FIT=1237 / VAL=308, SHA256 manifest đã lưu trong release.json;
nếu dùng toàn bộ FIT đó thì tổng text-training records là 1237+311=1548.
Không coi split LLM VAL=309 cũ là cùng manifest. Nhóm huấn luyện phải pin đúng
manifest; không tự tính lại/chọn split dựa trên chất lượng validation.

T3/T4 sửa audio_text nhưng giữ audio gốc: chỉ dùng text augmentation, không
dùng làm cặp WAV–transcript chuẩn để train ASR/audio model. Audio/state không
được bundle trong release Git này; nếu replay, dùng state synthetic DATA đã
bàn giao và runtime/catalog đúng phiên bản, giữ các ID còn lại.

## Các mẫu loại

`excluded_samples.csv` có đầy đủ 20 ID và lý do. Hai semantic-review findings:

- synth_t2_00014: partner Kevin Park phân biệt đối tác với hàng xóm nhưng nhãn vẫn hỏi contact_id.
- synth_t2_00047: Toàn trực ca phân biệt quản lý trực ca với bạn cùng lớp nhưng nhãn vẫn hỏi contact_id.

Các ID bị loại đã được bỏ khỏi file JSONL, không chỉ đánh dấu metadata. Không
thay nhãn hoặc sửa nội dung các record giữ lại. Không lấy pool 496 cũ merge lại
mà thiếu exclusion list, vì sẽ tái nhập các mẫu đã loại.

## Kiểm chứng và giới hạn

- Static checks: 311 mẫu, 0 error / 0 review / 0 advisory theo bộ rule hiện có.
- Replay mới: 254/254 mẫu có call khớp tool_results, final_state và call log;
  57 mẫu không call có nhãn rỗng hợp lệ.
- `release.json`: source/output hashes, counts, exclusions và split identity.
- `validation.json`: kết quả và các kiểm tra còn thiếu.

**Chưa có chứng nhận đúng ngữ nghĩa toàn bộ hoặc hiệu quả training.** Còn thiếu
adjudication độc lập toàn bộ, đối chiếu train gốc/leakage/provenance và đo VAL.
Đây là bản đã lọc để pipeline text-training có thể nạp và thử nghiệm, không phải
official dataset/submission hoặc tuyên bố model sẽ học tốt hơn.

## Tái tạo

Với JSONL 331 mẫu gốc tải từ folder DATA:

```sh
python3.11 data/releases/synthetic_phase1_v2_filtered_311/verify_release.py \
  --source /path/to/original/synthetic_phase1.jsonl \
  --output /path/to/new/filtered.jsonl
```

Verifier khóa SHA256 nguồn, áp exclusion list và so hash kết quả; từ chối ghi
đè output. Không download, train hoặc execute tools.
