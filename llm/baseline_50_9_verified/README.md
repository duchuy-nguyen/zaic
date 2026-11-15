# LifeMate baseline 50,9 — code để nhóm dùng chung

**PhoWhisper-large đã fine-tune LoRA + Qwen3.5-4B đã fine-tune QLoRA + helper ngày/giờ V1.**
Điểm **50,9 là điểm public do thành viên báo sau khi nộp BTC**, không phải lần chấm mới của gói code này.

Đây là folder code public. Weights, checkpoint, dữ liệu, bank retrieval, split IDs, reference, traces và JSONL kết quả nhận/chia sẻ riêng. Các manifest public chỉ chứa tên file, kích thước và checksum; không chứa trọng số hay nội dung mẫu. Không có API token hoặc link Drive private.

## 1. Pipeline

    Audio current + audio của mọi user turn trong history
      -> PhoWhisper-large đã merge LoRA FP16, greedy, batch2
      -> transcript gốc + context + assistant/tool history
      -> gợi ý số/giờ; không sửa thay transcript
      -> tối đa 1 ví dụ TRAIN-fit bằng word TF-IDF
      -> Qwen3.5-4B Q4_K_M, native function calling, 1 tool/bước
      -> ToolEnv.validate -> ToolEnv.call -> đọc observation thật -> bước kế
      -> ASK nếu còn thiếu thông tin / STOP khi hoàn tất
      -> id, tool_calls, state_hash do phiên ToolEnv thực tế cung cấp

History đã được môi trường áp dụng, không thực thi lại. Context/schema chỉ cho phép tool của app được cài. Không đọc hidden state hoặc labels public để suy luận. Mỗi mẫu một ToolEnv, giữ toàn bộ tool log kể cả lỗi; không reset/rerun chọn lọc.

| Phần | Bản đã chốt |
|---|---|
| ASR | vinai/PhoWhisper-large, LoRA r32/alpha64, 2 epoch/250 update; merge FP16 |
| LLM | Qwen/Qwen3.5-4B, QLoRA r16/alpha32, 1 epoch/489 update; merge GGUF Q4_K_M |
| Qwen thực deploy | 4.205.751.296 tham số; đủ32 text blocks, không thêm pruning |
| ASR thực deploy | 1.543.304.960 tham số |
| Tổng | 5.749.056.256 tham số logic; quantization không giảm số tham số |
| Dữ liệu | 1.236 TRAIN-fit /309 VAL theo nhóm; Qwen3.906 mẫu next-action TRAIN |
| Decode | Greedy, temperature0, thinking off, seed20261103; context16.384/output512 |
| Agent | Tối đa8 calls/2 schema repairs; 30giây/mẫu gồm ASR, chuyển/nạp model và agent |
| Placement | Serial ASR -> chuyển ASR về CPU -> llama-server -> unload; audio batch2 |

**So với50,2:** thêm helper ngày/giờ của50,5; đổi Medium sang trained Large và dùng serial batch2. Qwen vẫn là weights V2 cũ. Không chứa adapter observation/DPO hoặc các thử nghiệm về sau. Xem MODEL_IDENTITY.json và training_recipe.json.

## 2. Folder này có gì?

| File/thư mục | Chức năng |
|---|---|
| baseline50/actor.py | Native next-action loop, validate/call/observation/ASK/STOP |
| baseline50/asr.py | PhoWhisper greedy, WAV16kHz mono PCM16, current/history audio, kiểm EOS và thời hạn |
| baseline50/backend.py | Quản lý llama-server loopback; context/token/decode; key localhost sinh tạm và được che khi ghi command |
| baseline50/policy.py + temporal.py + prompt.py | TF-IDF1demo, hints số/ngày giờ và system prompt |
| baseline50/scoring.py | Scorer local gần công thức BTC, có giả định fuzzy; không phải scorer chính thức |
| baseline50/io.py + release.py | Config/checksum/sanitized input/resume/manifest của chính release này |
| baseline50/run.py | CLI check/VAL/public; xuất records/traces/summary/submission/description |
| assets.manifest.json | Metadata của bộ Large private chính xác cần nhận |
| MODEL_IDENTITY.json + SOURCE_PROVENANCE.json | Nhận diện model và đối chiếu nguồn/hash |
| training_recipe.json + training_reference/ | Recipe và mã training lịch sử; xem giới hạn tái lập trong README riêng |
| RUNTIME.json | Runtime serial batch2 đã chọn trên TRAIN |
| tools/ | Audit public tree và giải nén ZIP private an toàn |
| tests/ | CPU tests bằng dữ liệu giả; không cần weights/dataset/GPU |

Module Python vẫn tên baseline50 để giữ giao diện tích hợp của nhóm. Dùng venv riêng và chạy từ folder này để không import nhầm package50,2.

## 3. Setup inference

Môi trường gốc đã chạy: Windows11/NVIDIA, Python3.13.13, Torch2.8.0+cu128, Transformers4.57.1; llama.cpp CUDA b11800. Source dùng Python>=3.11; mới kiểm package/model theo các phiên bản ghi trên, không phải mọi phiên bản. Chạy từ source folder; không đóng gói wheel/pip install .

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    python -m pip install -r requirements-inference.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
    Copy-Item config.example.json config.local.json

Nhận bộ **Large50,9** private và dataset BTC; sửa bốn đường dẫn trong config.local.json. Các đường dẫn tương đối tính theo vị trí file config. Để assets/dataset/outputs ngoài checkout public là lựa chọn khuyến nghị. Không cần HF token hay API key bên ngoài khi đã có đủ assets.

    python -B -m baseline50 check --config config.local.json

Lệnh check đọc/hash tài nguyên và dataset; không tạo ToolEnv, không inference hay upload. File assets.manifest.json trong private_assets phải khớp nguyên bản metadata public; Medium hoặc model khác bị từ chối.

## 4. Chạy VAL/public và tiếp tục lượt đã lưu

    python -B -m baseline50 val --config config.local.json
    python -B -m baseline50 public --config config.local.json

Kết quả ở output_dir/val hoặc output_dir/public. Public đủ634 ID tạo `submission_baseline_50_9.jsonl` cùng `SOLUTION_DESCRIPTION.txt`. Không tự upload. Các file record/trace/score/ASR transcript chỉ để private.

Khi đã dừng ở ranh giới mẫu được commit:

    python -B -m baseline50 public --config config.local.json --resume

Tạo file `PAUSE_REQUESTED` tại gốc package để yêu cầu dừng sau mẫu hiện tại; chỉ bỏ cờ và resume khi muốn chạy tiếp. Nếu inflight chưa commit, giữ artifacts để điều tra; không xóa cờ inflight để chạy lại mẫu public.

Folder output phải mới cho bản package này. Không chép run_manifest của thí nghiệm gốc vào để resume: metadata/source hashes của lớp đóng gói khác. Không chạy lại public chỉ để chọn kết quả đẹp hơn.

## 5. Bàn giao cho nhóm

- Speech cung cấp merged Large đúng hash và processor.
- LLM: `actor.infer(view, catalog, env, backend)` nhận text/context/history và observation; không cần import training_reference.
- Ops dùng đúng assets, prompt, decode, helper và ToolEnv contract; quản lý hash/JSONL và đóng Docker.
- File này là baseline Windows có giao diện inference đã đóng gói, **không phải Docker submission hoàn chỉnh hoặc runner private đã kiểm chứng**. Không đổi runtime/Linux rồi khẳng định đã tái lập50,9 khi chưa kiểm tra.
- Điểm50,9 không bảo đảm tái lập trên mọi máy: deadline30s có thể gây kết quả khác. Cached-ASR DEV cũng không thay thế timing end-to-end.

## 6. Kiểm tra trước khi tự push

    python -B -m unittest discover -s tests -v
    python -B tools/audit_git_payload.py

Audit kiểm cả file đang bị Git ignore trong folder; nếu folder đã chứa assets/outputs thì phải tách chúng khỏi bản code trước khi xuất bản. Scan pattern không bảo đảm phát hiện mọi loại secret; hãy xem thêm danh sách staged.

Từ gốc repo, chỉ stage folder code này:

    git add -- llm/baseline_50_9_verified
    git diff --cached --stat
    git diff --cached --name-only

Không dùng git add . khi workspace còn nhiều thí nghiệm/dữ liệu. Commit message gợi ý: `feat(llm): add frozen LifeMate 50.9 baseline and setup docs`. Việc commit/push do người dùng thực hiện.

Xem PRIVATE_ASSETS.md để biết tài nguyên chia sẻ riêng. Baseline50,2 và thí nghiệm50,9 gốc được giữ nguyên.

## Bổ sung sau rà soát

Đây là bản đã rà soát để push. Folder trước giữ nguyên để đối chiếu, không phải bản nên stage. Phần inference không thay model/prompt/helper/decode. Mã training tham khảo đã tách đúng Qwen3.5 và PhoWhisper-large, không còn các nhánh Qwen3 hoặc Medium không dùng. Nó vẫn là source tham khảo cần workspace private, không phải training CLI được chứng nhận chạy độc lập.

Tên release trong assets.manifest.json có chữ baseline50.5-phowhisper-large-swap: đây là ID thí nghiệm lấy LLM/helper50,5 ghép trained Large và được thành viên báo50,9 sau khi nộp. Không đổi manifest gốc vì cần khớp assets private; MODEL_IDENTITY.json xác định rõ bản50,9 và hash Qwen/Large.
