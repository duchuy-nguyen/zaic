# Bàn giao baseline 50,9 cho Ops/Docker

## 1. Mục tiêu và trạng thái

Đóng gói đúng **trained PhoWhisper-large + trained Qwen3.5-4B + helper ngày/giờ V1 + TF-IDF1demo** để nhóm tích hợp Docker. Điểm50,9 là điểm public do thành viên báo của thí nghiệm gốc; package này không phải một lần chấm mới.

Package hiện đã kiểm source/hash và CPU tests. **Chưa có Linux runtime đã xác minh, Dockerfile, predict.sh, notebook đo thời gian hoặc private-test loader hoàn chỉnh.** Không copy nguyên CLI Windows vào Linux rồi coi là Docker đã chạy.

File này là hướng dẫn tích hợp từ code hiện có, không thay thế hướng dẫn BTC. Không thay weights bằng các checkpoint DPO/observation đang thử nghiệm.

## 2. Tài nguyên phải nhận riêng

Code Git không đủ để inference. Nhận bộ Large50,9 private theo [PRIVATE_ASSETS.md](PRIVATE_ASSETS.md).

| Thành phần | Tài nguyên / nhận diện |
|---|---|
| Qwen | models/qwen/Qwen3.5-4B-LifeMate-V2-Q4_K_M.gguf |
| Qwen SHA256 | cb09b88189a7a1e001022b08eac0e7609c560cd3dee0bca37f9414a85f74a8b1 |
| ASR | Toàn bộ models/asr/ của trained PhoWhisper-large đã merge FP16, gồm processor/tokenizer/config |
| Retrieval | private_data/retrieval_index.json, đúng bank TRAIN-fit |
| Split/groups | private_data/split.json, dùng loại source/group và warmup |
| Tham chiếu TRAIN | private_data/train_references.json chỉ cho chấm VAL; không chuyển vào actor |
| Tài nguyên warmup | Clip TRAIN đã chọn và metadata cần cho cách warmup gốc; không lấy public/private để thay tùy tiện |
| Mô hình/checksum | [MODEL_IDENTITY.json](MODEL_IDENTITY.json), [assets.manifest.json](assets.manifest.json) |
| Runtime Linux | Ops build/nhận riêng; Windows EXE/DLL trong bộ cũ không chạy trên Linux |

ASR có1.543.304.960 tham số; Qwen GGUF text-only có4.205.751.296, đủ32 text blocks; tổng5.749.056.256. Số này theo tensor logic, không chia cho số bit quantization. Không cần merge lại adapter khi đã có merged weights; không cần optimizer để inference.

Manifest assets public chỉ chứa tên/kích thước/hash. Giữ weights, bank ví dụ, split IDs, raw data, traces, results và link Drive có quyền truy cập ngoài GitHub. Chúng được bàn giao riêng cho nhóm và đưa vào Docker khi cần chạy offline.

## 3. Giữ nguyên chính sách inference

| Mục | Giá trị của baseline |
|---|---|
| ASR | FP16; language vi; task transcribe; greedy, beams1; max_new_tokens256; batch2 |
| Audio | WAV16kHz mono PCM16; current và mọi user turn trong history |
| Qwen | GGUF Q4_K_M; greedy temperature0/top_k1; seed20261103; thinking off |
| Context / output | 16.384 /512 token mỗi lượt; không truncate để ép vừa |
| Tool loop | 1 call/lượt, nhận observation thật rồi mới quyết định tiếp |
| Guard nội bộ | 8 calls, 2 schema/format repairs; đây là cấu hình baseline, không khẳng định BTC chỉ cho8 calls |
| Runtime | Serial ASR → chuyển ASR về CPU → nạp/chạy Qwen → giải phóng Qwen |
| Deadline | 30giây toàn mẫu; gồm ASR, chuyển/nạp model phát sinh trong mẫu, agent, tool và kết quả |
| ToolEnv | Một phiên thật/mẫu; output lấy call log và state_hash từ chính phiên đó |

ASR không fallback sang audio_text/labels. Không rút history, không chạy lại hành động history. Không đổi sang resident, tăng beam, bỏ tool lỗi, thêm default tham số hoặc thay prompt trong lúc port rồi gọi đó là cùng bản50,9. Nếu cần đổi execution profile trên GPU khác, tạo phiên bản có provenance và kiểm lại hành vi/timing.

## 4. Giao diện bên LLM

Các module inference cốt lõi không phụ thuộc training_reference.

### Nhận transcript từ Speech

`baseline50.asr.audio_view(row, transcripts)` trả view:

```python
{
    "id": "...",
    "latest_text": "...",
    "context": {...},
    "history": [
        {"role": "user", "text": "..."},
        {"role": "assistant", "text": "...", "tool_calls": [...],
         "tool_results": [...]}
    ]
}
```

row dùng whitelist id/audio/context/history. transcripts map đường dẫn audio sang ASR text, đủ current và mọi user-history audio. Đây là cấu trúc minh họa, không phải dữ liệu cuộc thi.

### Khởi tạo retrieval và gọi agent

```python
from pathlib import Path
import time
from baseline50 import actor

# Một lần khi khởi tạo process, từ assets private đúng checksum:
actor.bind_package(Path("/models/private_data"))

# Ops đã có đúng MỘT ToolEnv, backend tương thích và deadline toàn mẫu.
# Không tạo ToolEnv thứ hai trong hàm này.
def infer_text(view, catalog, env, backend, outer_deadline):
    remaining = outer_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("No remaining whole-sample budget")
    return actor.infer(
        view, catalog, env, backend,
        seconds=remaining, max_calls=8, max_repairs=2,
    )
```

catalog là dict tên tool → spec của đúng split. env cung cấp validate(name, arguments), call(name, arguments), get_tool_calls(), get_state_hash().

actor đã tự xây prompt, lọc tool theo installed_apps, thêm hints và chọn demo. Không enrich input thêm một lần ở Ops. actor.infer **đã sở hữu vòng lặp validate/call/observation**; không đặt nó vào một harness khác cũng execute tool. Hợp đồng respond(ModelRequest) của scaffold Ops không phải giao diện generate dưới đây; cần adapter nếu dùng scaffold.

Kết quả actor gồm id/tool_calls/state_hash/status/error/elapsed_s/repairs/trace. finished hoặc asked_missing_info chỉ là trạng thái dừng, không chứng minh task đúng. elapsed_s của actor chỉ đo phần text-agent; không dùng thay thời gian end-to-end.

### Hợp đồng backend Qwen

`backend.generate(messages, tools, deadline, max_tokens=512) -> (response, prompt_tokens)`.

- response giữ cấu trúc native choices/message/tool_calls và usage.prompt_tokens mà parser hiện tại nhận.
- Backend cần /apply-template, /tokenize và /v1/chat/completions tương thích. Kiểm token theo template thực; không ước lượng bằng số ký tự.
- Prompt +512 reserve phải ≤16.384; response không có reasoning và không bị cắt vì token cap.
- Payload/flags chuẩn xem baseline50/backend.py: parallel_tool_calls=False, temperature0, top_k1, enable_thinking=False, seed20261103.
- Chỉ loopback cho server; key localhost sinh tạm và phải che khỏi command/log được chia sẻ. Không dùng API inference bên ngoài.

## 5. Phần Ops cần port cho Linux/private

### A. Runtime và manifest theo platform

baseline50/backend.py hiện gọi Windows llama-server.exe. baseline50/io.verify_assets kiểm toàn bộ asset manifest, gồm hash Windows runtime. **Chỉ sửa đường dẫn executable trong config là chưa đủ.**

Ops cần build/nhận Linux llama-server hỗ trợ đúng GGUF Qwen3.5; ưu tiên đối chiếu source/tag b11800 của runtime gốc, ghi commit/build flags/CUDA và hash binary. Model weights, processor, prompt, helper và decode vẫn giữ nguyên.

Tạo manifest và verifier cho phiên bản Docker/Linux, giữ các hash model/bank gốc và thêm hash runtime Linux. Không bỏ kiểm tra hash để ép chạy. Giữ source Windows làm mốc đối chiếu; đánh dấu rõ những thay đổi platform. Assets manifest nguyên bản không được sửa rồi coi là bản gốc.

Inference Python gốc: Torch2.8.0+cu128 /Transformers4.57.1. Không cài Transformers5.17.0 của training vào môi trường inference. requirements-inference.lock.txt là bằng chứng môi trường gốc; stack wheel Linux/CUDA cần build và kiểm lại, không phải lock Docker đã xác nhận.

### B. Loader dữ liệu

CLI hiện chỉ nhận check/val/public. load_inputs khóa634 public IDs cùng catalog/SDK hash v3.2.0, nên **không phải private runner tổng quát**.

Ops viết loader split mount thực tế: tìm JSONL/catalog/zaic_tools của split, whitelist input, import SDK đúng split trong process riêng và tạo ToolEnv từ root mount. Không ép private SDK dùng fingerprint public. Không đọc trực tiếp thư mục environment hoặc truyền state/labels vào actor.

Bản Windows còn dùng TRAIN cho warmup và verify toàn bộ manifest, kể cả train_references. Nếu làm container inference tối giản, cần xử lý rõ phần startup/warmup và kiểm tài nguyên tương ứng; không xóa file rồi vô hiệu hóa verifier. train_references không phải input model.

### C. Budget và output

Caller giữ một deadline monotonic chung từ các bước xử lý case. Truyền phần thời gian còn lại vào ASR, model startup và actor; không cấp lại30giây cho từng stage hoặc từng tool. Kiểm tổng thời gian sau lấy hash/chuẩn bị output.

Writer lấy đúng ba trường id/tool_calls/state_hash từ kết quả phiên ToolEnv thật, giữ cả lỗi/timeout trong coverage. Không tự bịa hash, lọc call, reset phiên hoặc rerun riêng mẫu lỗi để chọn kết quả khác. Nếu execution/inflight chưa commit, giữ journal để điều tra.

Đầu ra của Windows CLI nằm tại output_dir/public/submission_baseline_50_9.jsonl. Docker adapter phải xuất đúng file BTC yêu cầu; không chỉ để nguyên file trong thư mục records.

## 6. Đối chiếu yêu cầu Docker của BTC

Các mục sau được đối chiếu ngày15/11/2026 từ [Docker Guide chính thức](https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf), mục4–8:

| Hạng mục | Đích cần bàn giao |
|---|---|
| Lệnh chạy | predict.sh nhận input mount /data, ghi /result/submission.jsonl |
| Notebook | predict_notebook.ipynb và start_jupyter.sh; có JupyterLab |
| Timing notebook | /result/jupyter_submission.jsonl, thêm time; kết quả phải tương đương submission |
| Hồ sơ | README phương pháp, training_code và dữ liệu/tài nguyên tái lập |
| Image | Archive .tar.gz, MD5, link tải được |
| Inference | Offline |

PDF này ghi RTX5090 32GB, driver595.58.03, CUDA13.2; tài liệu cũ local ghi A30. Ops đối chiếu nguồn BTC đang áp dụng trước khi khóa image, không lấy máy local làm chứng nhận.

## 7. Layout triển khai đề xuất của nhóm

Đây là layout đề xuất để nối code, không phải file đã có trong package:

```text
/code/
  baseline50/                 # core và adapters platform đã review
  predict.py                  # loader split + orchestration + writer của Ops
  predict.sh
  start_jupyter.sh
  predict_notebook.ipynb
  training_code/              # bundle tái lập riêng, đầy đủ
  README.md
/models/
  qwen/                       # merged GGUF gốc
  asr/                        # merged Large và processor
  private_data/               # retrieval/split/resources cần cho inference
/data/                        # input BTC mount
/result/                      # output BTC thu
```

training_reference trong repo **không tự thay thế training_code tái lập đầy đủ**: nó là source đã chọn nhánh để audit, còn cần manifests/data/snapshots/dependencies và quy trình train/export riêng. Đọc [training_reference/README.md](training_reference/README.md), [training_recipe.json](training_recipe.json) và [REVIEW.md](REVIEW.md).

## 8. Tiêu chí nhận bàn giao trước khi gọi Docker sẵn sàng

- [ ] Đúng Qwen/Large hashes; không lấy Medium, Qwen3 hoặc adapter thử nghiệm mới.
- [ ] Linux backend parse native tools, token counts, context và thinking giống contract.
- [ ] Loader chạy đúng SDK/split mount; không gắn public634 vào private.
- [ ] Một ToolEnv/case; không double-execute do hai agent harness.
- [ ] ASR đủ current/history; không gold fallback hoặc cắt audio/history.
- [ ] Đo deadline toàn mẫu, model-load và notebook timing; kiểm timeout/call/hash/coverage.
- [ ] Chạy image sạch offline với assets bên trong; không dựa cache hoặc venv của máy build.
- [ ] JSONL hai luồng predict/notebook hợp lệ, đủ ID và kết quả tương đương; giữ lỗi trong báo cáo.
- [ ] Đủ training bundle/README và tài nguyên tái lập riêng; không coi source tham khảo là đã tái lập.
- [ ] Ghi container/runtime digest, model hashes, source/config, MD5 archive và xác nhận nộp.

Đây là checklist cần Ops thực hiện, chưa phải xác nhận đã pass. Package hiện chỉ có bằng chứng source/hash/CPU tests; chưa build hay chạy Docker trong lần bàn giao này.
