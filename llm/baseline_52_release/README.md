# LifeMate — bản chốt public 52

Đây là bộ bàn giao mới, gom code inference, hướng dẫn tích hợp Docker và cả hai giai đoạn huấn luyện. Điểm 52 do thành viên báo sau khi nộp; không phải điểm được đo lại khi đóng gói. Bản 50.9 và mọi checkpoint cũ được giữ nguyên.

## Gửi gì cho ai?

- **GitHub public:** chỉ folder `baseline_52_release` này, đặt trong `llm/` của repo nhóm. Không có weights, audio, dữ liệu mẫu, split IDs, trace, kết quả nộp hoặc link Drive riêng.
- **Drive PRIVATE:** nguyên folder `LifeMate_52_DRIVE_PRIVATE_20261116`, gồm 5 ZIP và các file hướng dẫn/checksum/inventory. Chỉ chia sẻ trong đội.
- Người làm Docker dùng bộ inference và weights52 ngay; không cần train lại. Training được bàn giao để tái lập và hoàn thiện phần training_code của bài nộp.
- Bộ mới tự chứa các tài nguyên đã liệt kê; không cần ghép code từ nhiều thí nghiệm cũ. Không xóa các bộ50.9 đang lưu làm dự phòng.

## Cấu trúc code

| Đường dẫn | Mục đích |
|---|---|
| inference/ | CLI check/VAL/public; actor, ASR, prompt, backend, temporal helper và scorer local |
| training/base50_9/ | Chuẩn bị dữ liệu, train Qwen SFT/PhoWhisper-large, merge/export/quantize và tạo retrieval |
| training/dpo/ | Rebuild preference từ TRAIN rollout đã lưu, review/filter, reference, smoke, DPO+SFT replay và merge/export |
| DOCKER_HANDOFF.md | Đúng model52, giao diện actor/backend, phần phải port Linux và giới hạn kiểm chứng |
| TRAINING_REPRODUCTION.md | Thứ tự tái lập base → 50.9 → 52, input/output và lệnh đầy đủ |
| RELEASE_IDENTITY.json | Revision base, adapter/GGUF/submission checksum và tổng tham số |
| DRIVE_ARCHIVES.json | SHA256 và dung lượng 5 ZIP riêng |
| release_tools.py | Audit code public và giải nén/kiểm toàn vẹn tài nguyên private |
| VALIDATION.json | Những kiểm thực sự đã chạy; không mặc định Docker/full retrain đã pass |

Module inference vẫn có tên `baseline50` để giữ import/API; release/model đã là52. Không dùng weights V2 cũ cho manifest52, không sửa hash guard để ép chạy.

## Nhận tài nguyên từ Drive

Chạy từ folder chứa README này. Các đường dẫn đích nên ở ngoài checkout.

```text
python -B release_tools.py audit
python -B release_tools.py unpack --archive PATH/01_INFERENCE_MODELS_52.zip --dest PATH/inference52_assets
python -B release_tools.py unpack --archive PATH/02_RUNTIME_WINDOWS.zip --dest PATH/inference52_assets
python -B release_tools.py unpack --archive PATH/03_TRAINING_BASE50_9.zip --dest PATH/training_base50_9
python -B release_tools.py unpack --archive PATH/04_TRAINING_DPO52.zip --dest PATH/training_dpo52
python -B release_tools.py unpack --archive PATH/05_EVALUATION_AND_SUBMISSION.zip --dest PATH/evaluation52
```

ZIP01/02 giải nén chung một thư mục inference; ZIP03/04/05 giải nén riêng. Công cụ kiểm outer SHA và từng member, chặn đường dẫn thoát thư mục/symlink và không đè file khác hash.

Để chạy inference Windows: cài môi trường theo `inference/requirements-inference.lock.txt` trong venv riêng; copy `inference/config.example.json` thành `config.local.json`, sửa đủ bốn đường dẫn. Dataset/SDK BTC train+public_test đúng v3.2.0 do thành viên đã có quyền truy cập tự lấy; raw public_test không nằm trong bộ Drive.

```text
cd inference
python -B -m baseline50 check --config config.local.json
python -B -m baseline50 val --config config.local.json
python -B -m baseline50 public --config config.local.json
```

Public mới tạo `output/public/submission_baseline_52.jsonl` theo output_dir đã cấu hình. File đã nộp đạt52 nằm nguyên bản trong ZIP05: `public/submission_qwen35_decision_dpo.jsonl`, SHA256 trong RELEASE_IDENTITY. Không coi việc chạy lại ở máy khác là mặc định ra cùng bytes hoặc điểm.

## Model và cách huấn luyện

1. Qwen3.5-4B base revision cố định → QLoRA SFT 489 updates, 3906 mẫu next-action TRAIN → adapter V2 của50.9.
2. Adapter V2 → step DPO63 updates, 63 preference đã review, 189 mẫu SFT replay đi kèm. Reference là parent50.9, LR2e-6/beta0.1/dropout0; không có VAL/public gradient data.
3. PhoWhisper-large LoRA r32/alpha64, 3997 clip TRAIN, 2epoch/250updates, best theo generated VAL WER; giữ nguyên giữa50.9 và52.
4. Merge adapter vào đúng base, ASR FP16; Qwen merge BF16 rồi llama.cpp b11800 `--no-nextn`, Q4_K_M. Đủ32 text blocks, không pruning thêm. Tổng deployed5,749,056,256 tham số.

Tên/revision/URL/hash từng file base và seed/hyperparameters có trong training/base50_9. Không nhét thêm hàng chục GB model base vào Drive; URL HuggingFace ghim revision và hash cho phép lấy đúng bản. Các mã nguồn/chốt dữ liệu gốc và checkpoint cuối DPO có trong ZIP riêng.

## Kiểm chứng và giới hạn

- Bản thử nghiệm gốc đã chạy634public,0timeout và audit output đạt; điểm52 do người dùng báo. Weights/checkpoints gốc được giữ.
- Bảy module inference quyết định/ASR/backend/policy/prompt/scoring/temporal được giữ byte-identical với lượt chạy đó. Wrapper chỉ đổi cách xác minh package, metadata/nhãn và tên file output.
- Bộ portable mới được kiểm CPU, checksum và parity dữ liệu theo VALIDATION. Không tự chạy lại full training hoặc inference để đóng gói.
- Chưa build/test Linux Docker. Runtime ZIP02 là Windows .exe/DLL; Linux phải build runtime riêng và tạo manifest nền tảng mới có provenance, giữ nguyên model hashes. Đọc DOCKER_HANDOFF trước khi tích hợp.

## Push an toàn

Copy nguyên folder này vào `llm/` của checkout repo nhóm, chạy audit rồi chỉ stage đúng đường dẫn:

```text
python -B llm/baseline_52_release/release_tools.py audit
git add -- llm/baseline_52_release
git diff --cached --stat
git commit -m "feat(llm): add public52 inference and training handoff"
git push origin feat/llm-dog
```

Kiểm branch/remote là repo nhóm trước khi dùng lệnh. Nếu diff hiện `(END)` thì nhấn q. Không dùng `git add .` ở workspace thí nghiệm. Không tự động push/upload/submission từ code này.
