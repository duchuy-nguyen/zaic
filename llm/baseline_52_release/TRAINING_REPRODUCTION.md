# Tái lập base → 50.9 → public52

## Môi trường và model base

Recipe lịch sử trên Windows, Python3.13.13, RTX4060 Laptop BF16; torch2.8.0+cu128, transformers5.17.0, peft0.21.2, accelerate1.15.0, bitsandbytes0.48.2, FLA0.3.2 và Triton Windows3.4.0.post21. Xem training/base50_9/ENVIRONMENT.json và requirements. Tạo venv riêng, không thay package trong môi trường inference của đội. Linux cần wheel CUDA/Triton tương ứng và kernel tests; full retrain portable chưa được xác nhận trên Linux/T4.

Base model đúng revision và URL đã ghi ở `training/base50_9/BASE_MODELS.json`: Qwen/Qwen3.5-4B và vinai/PhoWhisper-large. Snapshot checks kiểm SHA256 từng file, không chọn model chỉ theo tên folder.

## 1. Khôi phục tài nguyên

Dùng `release_tools.py unpack` ở README: ZIP01+02 → inference52_assets; ZIP03 → training_base50_9; ZIP04 → training_dpo52. Base snapshots tải theo URL pin hoặc dùng bản đã có; không nằm trong ZIP. ZIP05 chỉ để kiểm kết quả, không dùng cho train.

ZIP03 có exact3906/963 SFT rows,3997/1039 audio-text clips, raw TRAIN annotation/catalog/audio, group split, chỉ TRAIN-fit noise mappings, token caches, parent Qwen adapter và ASR best adapter, processors, converter/quantizer. Split trước khi mở rộng:1236TRAIN-fit/309VAL. VAL không gradient nhưng dùng selection/development.

## 2. Huấn luyện parent Qwen SFT và ASR Large

Trong `training/base50_9`, copy config.example.json ra config.local.json, sửa private_dir trỏ training_base50_9, work_dir mới và hai base snapshot. Các lệnh:

```text
python -B reproduce.py verify --config config.local.json
python -B reproduce.py prepare --config config.local.json
python -B reproduce.py retrieval --config config.local.json
python -B reproduce.py profile --config config.local.json
python -B reproduce.py train-qwen --config config.local.json --smoke
python -B reproduce.py train-qwen --config config.local.json
python -B reproduce.py train-asr --config config.local.json --smoke
python -B reproduce.py train-asr --config config.local.json
python -B reproduce.py merge-qwen --config config.local.json
python -B reproduce.py merge-asr --config config.local.json
python -B reproduce.py export --config config.local.json
```

Merge mặc định dùng adapter lịch sử đã bàn giao; để dùng adapter mới train cần `--adapter-dir` và processor_dir tương ứng như README bên trong. Không nhầm adapter vừa train lại với adapter bit-identical lịch sử. Output Qwen ở work/qwen/outputs/llm_full/adapter_final; ASR work/asr/outputs/asr_full/best_adapter; merged ASR FP16, Qwen BF16→Q4_K_M.

Qwen parent489updates, r16/alpha32/dropout0.05, LR1e-4, batch1/acc8, native completion-only loss. ASR Large250updates/2epochs,r32/alpha64/dropout0.05, LR5e-4,batch1/acc32,warmup30; best generated VAL WER9.58145% ở250. Actual trainer args đầy đủ nằm trong recipe. Config chuẩn bị chung lịch sử vẫn nhắc Medium; trainer ASR của bundle chỉ khóa Large và dùng cùng audio/text manifests. Không train Medium hay Qwen3 bằng bộ này.

## 3. Preference data cho DPO

ZIP04 cung cấp frozen plan192TRAIN sources thuộc6strata, source/group IDs, TRAIN SDK/state, raw sources, captured actual step inputs và gold-prefix audits. Code `dpo_core/prepare.py` giữ algorithm harvest/build, chuyển exclusions từ literals sang file private.

- Chỉ so sánh tại prefix model đã thăm và khớp gold; không gán mọi alternative schema-valid thành negative.
- Contrast continue/stop/ask/argument; có17 actual model error pairs và synthetic contrasts được ghi rõ.
- Draft64pairs/192replay. Semantic review loại1pair, actual training63pairs từ63sources và189replay;236sourceIDs active trong tổng exposure. Không ép thành64updates để khớp field steps lịch sử trong recipe draft.
- Review exclusion được bàn giao nguyên bản trong PAIR_REVIEW; không tự động tạo review approved mới khi thu thập lại model outputs khác.

Tạo config.local.json từ `training/dpo/config.example.json`. Chạy từ `training/dpo`:

```text
python -B reproduce.py verify-data --config config.local.json
python -B reproduce.py rebuild-data --config config.local.json
```

`rebuild-data` cần work mới, dùng captured TRAIN rollout đã bàn giao để tái tạo toàn bộ6file data. Nó so SHA256 từng file với lịch sử, giữ review63 và tạo freeze portable. Không sinh inference mới, không đọc public data. Tokenization dùng base Qwen snapshot đúng revision; max9253tokens, không truncation. Dùng frozen plan/rollout là cách tái lập chính xác dữ liệu đã train, không lấy lại ngẫu nhiên192sources khác.

Nếu không cần rebuild mà dùng nguyên processed data, thay `rebuild-data` bằng `setup`. Hai lệnh đều yêu cầu work chưa tồn tại, không chạy cả hai trên cùng work. Input SDK/state chỉ dùng audit TRAIN; tuyệt đối không đưa chúng vào actor public/private.

Lệnh `harvest` có để thu thập lại từ parent model và actual ToolEnv trên192TRAINsources, nhưng work setup mặc định đã có historical harvest. Muốn thử rollout mới phải giữ bản lịch sử, dùng workspace mới và lập review/version mới; không ghi đè các file đã chốt. Source selection lịch sử và source gốc đầy đủ còn trong ZIP04/historical_source và SOURCE.json. Không cần harvest lại để tái lập từ bộ dữ liệu đã xử lý.

## 4. Reference, smoke và DPO training

```text
python -B reproduce.py math --config config.local.json
python -B reproduce.py reference --config config.local.json
python -B reproduce.py smoke --config config.local.json
python -B reproduce.py train --config config.local.json
```

Reference tính chosen/rejected log-probability từ parent SFT adapter trong ZIP03, NF4/BF16/dropout0. Nếu giữ nguyên đúng stack lịch sử, có thể dùng `restore-reference` thay `reference` để khôi phục cache đã bàn giao. Thay model/tokenization/precision stack thì tính lại, không reuse cache tùy tiện.

Full starts from parent adapter, không từ smoke.63updates,r16/alpha32,LR2e-6,beta0.1,3originalSFT replay/step,warmup2,clip1,AdamW CPU FP32 moments. DPO dùng exact chain-rule chosen/rejected backward riêng, không optimizer update giữa các pass. Saved activations offload CPU. Sau OOM lịch sử, allocator smoke/train được đặt `backend:cudaMallocAsync`; objective/data/order không đổi. Reference và inference không dùng override này.

Checkpoint giữ ở work/outputs/decision_dpo/checkpoints, mỗi10updates và1/final, gồm adapter+optimizer+scheduler+RNG+trainer_state. Chỉ `--resume` checkpoint do chính wrapper portable và cùng source/config tạo. Final checkpoint63 gốc trong ZIP04 là bản bảo toàn/đối chiếu, không hướng dẫn ép resume bằng đổi hash config; các checkpoint trước còn nguyên trên máy gốc.

Output mới: work/outputs/decision_dpo/adapter_final và tokenizer. Không có lời hứa train lại trên phần cứng/OS khác ra cùng hash hoặc điểm.

## 5. Merge/export bản52 mà không train lại

```text
python -B reproduce.py merge-export --config config.local.json
```

Lệnh dùng chính final adapter52 và tokenizer đã đóng ZIP04, merge Qwen BF16 trên CPU rồi converter/quantizer b11800 từ ZIP03. Output nằm work/final52_export. Tên GGUF của utility base có thể còn LifeMate-V2: nhận diện bằng adapter/weights SHA trong provenance, không theo tên. Model52 gốc đã dùng để đạt điểm nằm ZIP01, không cần tái xuất để dựng Docker.

Để merge adapter mới train, dùng utility training/base50_9 với `--adapter-dir PATH/new_adapter` và processor/tokenizer đúng run, work riêng. Với Linux quantizer phải pin đường dẫn/SHA binary Linux trong config, không chạy Windows .exe.

## 6. Retrieval và nguồn kiểm chứng

Retrieval TRAIN-fit TF-IDF1shot được tạo ở parent prepare, loại same source/group và giới hạn tools/visible IDs. DPO giữ nguyên index đó; `retrieval` kiểm byte hash. Không tạo lại index từ VAL hoặc public.

SHA và URL base nằm trong code; SHA từng archive/member nằm trên Drive; manifests/raw generation code/recipe/semantic review nằm trong private bundle. ZIP05 có local pairedVAL, public634traces, summary và file JSONL đã nộp, không có đáp án public. JSONL gốc52 SHA256: d29fde98949835eba78a467fe676c6dd51be69a06b5c6f12330ac11dd3597db7.

VALIDATION.json tách rõ CPU/data parity đã kiểm khỏi full training/GPU merge/Docker chưa chạy lại. Mọi artifact gốc được giữ; wrapper chỉ rebind private paths và chuyển guards phụ thuộc máy cũ sang package/data/base pins được ghi rõ.
