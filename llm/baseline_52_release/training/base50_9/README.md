# Bổ sung training cho baseline LifeMate 50.9

Folder này đi cùng `baseline_50_9_verified` (inference). Chỉ chứa code và metadata để đưa lên GitHub. Dữ liệu, audio, token cache, adapter và lịch sử train ở ZIP PRIVATE riêng; không đặt ZIP trong Git. Không có thao tác upload tự động.

## Đúng bản nào?

- Qwen3.5-4B V2: 3.906 mẫu next-action TRAIN, 489 optimizer updates, QLoRA r16/alpha32. Adapter SHA256 `7d04b7fd5feea6b92c230f4e710df25046a710c5d80ac9d8fe1f4e6dd73c93a5`.
- PhoWhisper-large: 3.997 audio TRAIN, LoRA r32/alpha64, 2 epoch/250 updates. Chọn best theo generated VAL WER: bước250, WER9.58145% trên1.039 audio VAL. Adapter SHA256 `fed80cdce5969ac72b5eae98b4d4fe28bccc9fccae2d39db093e4aa800502662`.
- Split theo nhóm trước khi mở rộng:1.236 TRAIN-fit/309 VAL. Qwen963 mẫu VAL không được gradient update. VAL có dùng để phát triển/chọn checkpoint, không phải blind test.
- 50.9 dùng temporal helper của bộ inference và ASR Large serial batch2. Không có DPO, observation LoRA, Qwen3 hay ASR-medium được train bằng các lệnh ở đây.
- `BASE_MODELS.json` có tên, URL revision cố định và SHA256 từng file base. Không cần tải lại model nếu đã có đúng snapshot.

## Bộ private cần nhận

`LIFEMATE_50_9_TRAINING_PRIVATE.zip` gồm:

| Đường dẫn sau giải nén | Nội dung |
|---|---|
| prepared/ | Đúng train/val SFT, audio-text manifests, split/group IDs, retrieval bank và prompt/config đã dùng |
| raw/ | Annotation TRAIN gốc, catalog và audio dùng cho TRAIN/VAL; không có public test |
| augmentation/ | Chỉ transcript noise của TRAIN-fit đã dùng; không có output LLM hoặc VAL/public predictions |
| tokens/ | Token caches Qwen và token profile gốc |
| adapters/, processors/ | Đúng hai adapter đã merge ra baseline50.9 và tokenizer/processor |
| history/ | Source gốc và manifests chứng minh quá trình train/merge/export; chỉ giữ private |
| converter/ | Source llama.cpp b11800 đúng hash, gồm license |
| runtime_windows/ | Quantizer và DLLs gốc, chỉ chạy Windows |

Config lịch sử trong prepared/ vẫn ghi ASR-medium vì nó mô tả lần chuẩn bị chung ban đầu. Hai danh sách audio/text của lần này đã được dùng nguyên vẹn để train Large. Lệnh `train-asr` khóa PhoWhisper-large theo `training_recipe.json`, không đọc cấu hình Medium để train. Source gốc private của ASR có nhánh Qwen3 lịch sử; wrapper công khai chỉ dùng hàm ASR đã tách.

## Cài môi trường

Môi trường lịch sử: Windows, Python3.13.13, RTX4060 Laptop BF16, torch2.8.0+cu128. Xem `ENVIRONMENT.json` và file requirements. Tạo venv riêng, cài torch từ index CUDA12.8 rồi cài các package còn lại theo pin. Qwen dùng FLA0.3.2 và Triton Windows3.4.0.post21. Không đổi package trong môi trường inference đang chạy.

```text
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements-training.txt
```

Đây là lệnh Windows. Linux Docker cần wheel Triton/CUDA phù hợp nền tảng; giữ phiên bản toán học và chạy kernel probe trước khi train. Bộ này không phải Docker image và chưa xác nhận một lượt train lại trên Linux. T4 không hỗ trợ BF16 nguyên sinh và không phù hợp với guard của recipe gốc.

## Chạy theo thứ tự

Các lệnh dưới đây chạy từ folder chứa file này. Sao chép `config.example.json` thành `config.local.json`, sửa đường dẫn PRIVATE/work và hai snapshot base. Đặt private và work ngoài checkout; dùng work mới để tránh đè dữ liệu.

```text
python -B reproduce.py unpack --archive PATH/LIFEMATE_50_9_TRAINING_PRIVATE.zip --checksums PATH/SHA256SUMS.json --dest PATH/private_training
python -B reproduce.py verify --config config.local.json
python -B reproduce.py prepare --config config.local.json
python -B reproduce.py retrieval --config config.local.json
python -B reproduce.py profile --config config.local.json
python -B reproduce.py train-qwen --config config.local.json --smoke
python -B reproduce.py train-qwen --config config.local.json
python -B reproduce.py train-asr --config config.local.json --smoke
python -B reproduce.py train-asr --config config.local.json
```

- `prepare` tạo `work/prepared_rebuilt`. Phải khớp SHA256 tám input thực tế, gồm train/val SFT, audio manifests và retrieval. `DATA_MANIFEST` mới có metadata khác vì đường dẫn/source portable và noise đã lọc TRAIN; không sửa file lịch sử để giả khớp hash.
- `profile` tạo `work/qwen/outputs/{train,val}_tokens.jsonl`, đối chiếu token hashes lịch sử, không cắt context. TRAIN max9.932tokens.
- Qwen output: `work/qwen/outputs/llm_full/adapter_final`; processor ở `llm_full/processor`. ASR: `work/asr/outputs/asr_full/best_adapter`, processor ở `asr_full/processor`.
- Smoke chạy2updates độc lập, không phải parent cho full train. Full bắt đầu từ base. Chỉ thêm `--resume` để tiếp tục checkpoint do chính wrapper mới này tạo; không resume checkpoint gốc vì source/provenance khác.
- Qwen giữ checkpoint mỗi10updates; ASR mỗi10 và cuối epoch, gồm optimizer/scheduler/RNG. Không xóa các checkpoint gốc trên máy chủ sở hữu. Khi chuyển máy cần copy đầy đủ checkpoint, không chỉ adapter.
- Không dùng label public/private làm dữ liệu train. Noisy transcripts ở đây là đầu vào tăng cường TRAIN, không phải nhãn thay thế gold tool calls. App masking chỉ khi đủ điều kiện, không ép tỷ lệ; source gốc có0 mẫu hợp lệ.

## Merge và xuất weights

Để dựng lại từ **adapter lịch sử đã bàn giao**, không cần train lại:

```text
python -B reproduce.py merge-qwen --config config.local.json
python -B reproduce.py merge-asr --config config.local.json
python -B reproduce.py export --config config.local.json
```

Để merge adapter vừa train mới, truyền `--adapter-dir PATH/adapter` và khai báo `qwen_processor_dir` hoặc `asr_processor_dir` trong config. Với ASR chọn best theo VAL WER, không tự mặc định final khi best khác final. Dùng work mới nếu đã merge/export trước đó.

Output: `work/merged/qwen`, `work/merged/asr` (FP16) và `work/exports/Qwen3.5-4B-LifeMate-V2-Q4_K_M.gguf`. Converter pin b11800, `--outtype bf16 --no-nextn`, Q4_K_M; giữ full32text blocks, converter bỏ phần vision/MTP không deploy. Qwen merge trên CPU BF16, ASR trên CPU FP16. Không tải model khi inference.

Trên Windows dùng quantizer/DLL trong ZIP. Linux cần quantizer b11800 đã build cho Linux; cấu hình đường dẫn và SHA256 riêng, không đặt file .exe vào Linux image. Thay binary/nền tảng không được mặc định là byte-identical với GGUF lịch sử. GGUF50.9 gốc SHA256: `cb09b88189a7a1e001022b08eac0e7609c560cd3dee0bca37f9414a85f74a8b1`.

## Giới hạn xác nhận

Các hàm train được trích từ source lịch sử, giữ kiến trúc/target/loss/hyperparameters. Wrapper mới đổi root dữ liệu/base/output và thay guard phụ thuộc máy cũ bằng kiểm toàn vẹn code + private payload + split + base pin; mọi thay đổi ghi ở `PROVENANCE.json`. Bộ gốc được giữ nguyên. Không bảo đảm train lại trên GPU/OS khác sẽ ra hash hoặc điểm giống hệt.

Việc bàn giao không khởi động một lần train mới hay build Docker. `VALIDATION.json` ghi các kiểm thực sự đã chạy, tách riêng kiểm CPU/data parity khỏi kiểm GPU/full retrain chưa chạy. Muốn chạy bản50.9 ngay, dùng weights inference gốc trong bộ Drive50.9 và `baseline_50_9_verified/DOCKER_HANDOFF.md`.

## Push / Drive

- GitHub: chỉ folder `baseline_50_9_training_repro` này, cạnh `baseline_50_9_verified` đã có. Chạy `python -B audit_public.py` trước.
- Drive PRIVATE: toàn bộ folder `LifeMate_50_9_TRAINING_PRIVATE_20261115`, bổ sung bên cạnh folder inference cũ. Không thay hoặc xóa folder inference cũ.
- ZIP có dữ liệu cuộc thi và adapter: chỉ chia sẻ trong đội, không đưa vào repo public.
