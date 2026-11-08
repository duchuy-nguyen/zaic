# LifeMate baseline — public 50,2

Baseline để nhóm dùng chung: **PhoWhisper-medium LoRA mới + Qwen3.5-4B QLoRA của bản38,6**. Điểm public **50,2 do thành viên báo sau khi tự nộp BTC**; gói đóng lại này không phải một lần nộp mới.

**Chỉ đưa thư mục code này lên GitHub. Trọng số, checkpoint, bank ví dụ, split, reference và kết quả mẫu nằm trong các ZIP Drive private riêng.** Không tự push/upload, không Git LFS cho weights trong baseline này.

## 1. Pipeline có gì?

```text
Audio hiện tại + toàn bộ audio user trong history
  → PhoWhisper-medium đã merge LoRA, FP16
  → text + context + history assistant/tool đã có
  → gợi ý số/giờ, giữ transcript gốc
  → tối đa 1 ví dụ TRAIN-fit tương tự bằng word TF-IDF
  → Qwen3.5-4B native tool calling: đề xuất 1 tool/lượt
  → ToolEnv.validate → ToolEnv.call → đọc kết quả thật → quyết định tiếp
  → hỏi thiếu thông tin / dừng
  → JSONL: id, tool_calls, state_hash
```

Các hành động trong history đã được thực hiện, không chạy lại. Chỉ tools của app được cài mới được đưa vào schema. ToolEnv thực thi và tính state hash; Qwen không tự viết hash. Không dùng API suy luận bên ngoài.

| Phần | Artifact của baseline50,2 |
|---|---|
| ASR | `vinai/PhoWhisper-medium`, revision55a7e3eb6c906de891f8f06a107754427dd3be79; LoRA r32/alpha64, attention+fc1/fc2, 2epoch,250/250steps; merge FP16 |
| LLM | `Qwen/Qwen3.5-4B`, revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; QLoRA NF4/BF16 r16/alpha32, attention/linear-attention/MLP,1epoch/489steps; inference GGUF Q4_K_M |
| Dữ liệu train |1.236 hội thoại TRAIN-fit;309 VAL ngoài gradient; Qwen mở rộng thành3.906 mẫu next-action TRAIN |
| Decode | Greedy, temperature0, thinking off, seed20261103; context16.384, tối đa512 output tokens/lượt |
| Agent | Tối đa8 tool calls,2 sửa định dạng/schema;30giây/mẫu gồm ASR live current/history, LLM, tools và hash |
| Tổng tham số đã merge |5.303.123.456, thuộc nhóm **Based on small models (<6B)** |

Giữ helper/retrieval/agent cũ đúng bản38,6; thay đổi tạo candidate50,2 là ASR. Gói này không chứa B+, evidence ledger hay Qwen A đang train dở.

## 2. Code được chia thế nào?

| File/thư mục | Chức năng |
|---|---|
| `baseline50/actor.py` | Vòng lặp Qwen→validate→execute→observation, ASK/STOP; giữ logic V2 |
| `baseline50/prompt.py` | SYSTEM prompt đã ghép đúng bản nộp |
| `baseline50/policy.py` | Giữ nguyên helper số/giờ và word TF-IDF lấy tối đa1 ví dụ TRAIN |
| `baseline50/backend.py` | Quản lý một llama-server loopback của job, native request, kiểm token/context; key localhost sinh tạm, không ghi key vào manifest |
| `baseline50/asr.py` | Model medium FP16, WAV16kHz mono PCM16, greedy batch4, ASR live cho current/history; không fallback sang text chuẩn |
| `baseline50/io.py` | Cấu hình đường dẫn, checksum, lock, ghi atomic, input allowlist và checkpoint từng mẫu |
| `baseline50/run.py` | CLI check/VAL/public; xuất traces/summary/submission/description; không upload |
| `baseline50/scoring.py` | Scorer local-spec-v1 có giả định fuzzy/exact keys; không phải scorer chính thức BTC |
| `assets.manifest.json` | Danh sách tên/dung lượng/SHA của assets private và fingerprint dataset/SDK; **chỉ metadata, không chứa weights** |
| `training_recipe.json` | Cấu hình huấn luyện và source provenance của hai model |
| `training_reference/` | Source huấn luyện/chuẩn bị dữ liệu lịch sử để đọc/audit. Phụ thuộc layout/kernels của project cũ; không phải CLI train portable |
| `tools/unpack_assets.py` | Giải nén ZIP Drive theo manifest/hash; giữ nguyên file khác đã tồn tại |
| `tools/audit_git_payload.py` | Kiểm cây code sạch, không weights/binary/raw data/credential pattern |
| `tests/` | CPU tests bằng dữ liệu giả: history, schema/repair/ASK, resume và archive safety |

Đường dẫn máy được đưa vào `config.local.json` (bị Git ignore), không sửa source code theo ổ đĩa từng người. Giao diện tích hợp theo vai trò: Speech dùng `MediumASR.transcribe`; LLM dùng `actor.infer(view,catalog,env,backend)`; Ops nối SDK và đóng gói runtime.

## 3. Nhận các gói Drive

Người chia sẻ đưa **4 ZIP và SHA256SUMS.json** lên Drive private của nhóm:

1. `LIFEMATE_BASELINE_50_2_MODELS_AND_PRIVATE_DATA.zip`: model ASR FP16 + processor, Qwen GGUF, retrieval bank TRAIN-fit, split và reference TRAIN để chấm VAL. **Bắt buộc cho inference**.
2. `LIFEMATE_BASELINE_50_2_RUNTIME_WINDOWS.zip`: llama.cpp CUDA b11800 và DLL/license tương ứng. **Bắt buộc cho cấu hình Windows mặc định**.
3. `LIFEMATE_BASELINE_50_2_TRAINING_CHECKPOINTS.zip`: đủ checkpoint ASR250/Qwen489 gồm LoRA, optimizer, scheduler, RNG, trainer state và data indexes/config. Dùng lưu trữ/khôi phục huấn luyện; **không bắt buộc để chạy inference**.
4. `LIFEMATE_BASELINE_50_2_ORIGINAL_SUBMISSION.zip`: JSONL gốc đã được người dùng báo50,2 + description/verification. Dùng đối chiếu; **không phải labels public**.

Không giải nén các ZIP vào thư mục code để commit. Dùng công cụ sau để hai ZIP assets cùng vào `private_assets/`; manifest của từng ZIP được lưu riêng, không ghi đè nhau:

```powershell
python tools/unpack_assets.py --archive "D:/Downloads/LIFEMATE_BASELINE_50_2_MODELS_AND_PRIVATE_DATA.zip" --checksums "D:/Downloads/SHA256SUMS.json" --dest private_assets
python tools/unpack_assets.py --archive "D:/Downloads/LIFEMATE_BASELINE_50_2_RUNTIME_WINDOWS.zip" --checksums "D:/Downloads/SHA256SUMS.json" --dest private_assets
```

Layout sau giải nén:

```text
private_assets/                    # Git ignore toàn bộ
├── models/asr/                    # model.safetensors + processor/config
├── models/qwen/...Q4_K_M.gguf
├── private_data/
│   ├── retrieval_index.json        # có text/ví dụ TRAIN, không push GitHub
│   ├── split.json
│   └── train_references.json       # chỉ scorer VAL đọc sau khi infer xong
└── runtime/llama-server.exe + DLLs
```

Dataset/audio/SDK chính thức **không được sao chép vào gói Git hoặc các ZIP model**. Mỗi thành viên tự nhận dataset BTC, giữ layout `train+public_test/train/` và `train+public_test/public_test/`, mỗi split có JSONL, tools catalog, audio và `zaic_tools/`.

## 4. Setup và chạy

Môi trường gốc đã chạy50,2: **Windows11 + NVIDIA CUDA, Python3.13.13, Torch2.8.0+cu128, Transformers4.57.1, llama.cpp b11800 CUDA12.4**. Các path đã portable giữa máy Windows. Linux/Kaggle/Docker chưa được xác nhận bằng gói runtime này; không tự đổi server/version rồi gọi đó là kết quả50,2 đã kiểm.

Chạy trong thư mục chứa README này, dùng môi trường Python riêng:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.lock.txt
Copy-Item config.example.json config.local.json
```

Sửa4 đường dẫn trong `config.local.json`: assets, dataset, outputs và llama-server. Mọi path tương đối được tính theo vị trí file config, không theo ổ đĩa của người tạo. Không cần Hugging Face token/API key để inference vì assets đã có local. Chạy check chỉ đọc/hash, không infer hay replay ToolEnv:

```powershell
python -m baseline50 check --config config.local.json
python -m baseline50 val --config config.local.json
python -m baseline50 public --config config.local.json
```

Public tạo `outputs/public/submission_baseline_50_2.jsonl` và `SOLUTION_DESCRIPTION.txt`. Người dùng tự nộp; **code không có chức năng upload**. `state_hash` lấy từ SDK thật. Không dựng lại public ToolEnv để audit/replay.

Không chạy hai job vào cùng output. Nếu gián đoạn sau khi một mẫu đã commit, dùng đúng code/assets/config và `--resume`. Nếu có mẫu **inflight chưa commit**, chương trình dừng để điều tra; không xóa marker/reset/rerun chọn lọc public. Sau khi sửa recipe/code/model, dùng output phiên bản mới.

## 5. Kết quả và giới hạn

| Pipeline | VAL local/100 | Tool F1 | State match | WER tổng |
|---|---:|---:|---:|---:|
| Medium V2 + Qwen V2 |45,91|59,79%|56,96%|21,39%|
| Medium V3 + Qwen V2 |**57,65**|67,86%|69,90%|**10,79%**|

VAL309: delta+11,75; grouped bootstrap CI95%[+6,84;+16,65];0 timeout/lỗi. Public gốc634:0 timeout/lỗi xử lý,4 mẫu chạm ngân sách8calls vẫn giữ nguyên; **50,2 do người dùng báo sau khi nộp**. Không có nhãn public để tự tính điểm. VAL đã dùng phát triển/chọn checkpoint, không phải blind test. Thời gian phụ thuộc máy và không được thay bằng thời gian đọc transcript cache.

Gói đóng lại kiểm checksum, parity prompt/schema/helper/ASR-decode/backend-request và CPU regression; không tự train hoặc rerun634 public chỉ để kiểm packaging. Điểm50,2 gắn với JSONL gốc có SHA trong `SOURCE_PROVENANCE.json`, không phải cam kết mọi môi trường đều sinh bit-identical outputs.

## 6. Chia sẻ GitHub và checkpoint

Trước khi người dùng tự add/commit thư mục code, chạy trên bản sạch chưa tải assets:

```powershell
python -B tools/audit_git_payload.py
python -B -m unittest discover -s tests -v
```

`.gitignore` chặn private_assets, model extensions, checkpoint, raw audio/JSONL, outputs, logs, venv và config.local. Audit payload cố ý từ chối nếu có assets private trong cây đang chuẩn bị chia sẻ; chạy trên bản code-only hoặc chuyển assets ra ngoài rồi trỏ config. Ignore không xóa file đã từng được track, vì vậy **người dùng cần xem danh sách staged trước khi tự push**. Không dùng force-add weights, không upload ZIP Drive vào GitHub.

Checkpoint ASR250/Qwen489 đã đủ epoch dự kiến, không replay bước251/490 khi chỉ finalize. Archive giữ đủ optimizer/scheduler/RNG; adapter một mình không phải full training resume. Các lần train tiếp theo nên có thư mục phiên bản riêng, lưu định kỳ/rút phiên tại optimizer boundary và ghi base revision, split, config, step, hash. Giữ nguyên baseline50,2 làm mốc đối chiếu.
