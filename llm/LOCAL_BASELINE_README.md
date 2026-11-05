# Chạy thử Qwen và Spark trên LifeMate bằng GGUF

Bộ chạy local để bắt đầu phần LLM. Python điều phối dữ liệu/tool; llama.cpp CUDA sinh kết quả qua HTTP **127.0.0.1 trong cùng máy**. Chưa ASR, chưa fine-tune, chưa Docker submission, không gọi API model bên ngoài và không tự cập nhật GitHub.

## 1. Có gì trong thư mục?

| Đường dẫn | Vai trò |
|---|---|
| Hai file `*.gguf` ngay trong LifeMate | Qwen3-4B-Instruct-2507 và Spark-X2.5-4B, Q4_K_M |
| `.runtime/b11800/` | llama.cpp CUDA 12.4 bản portable; runtime release 21/10/2026, không cài global |
| `data/lifemate-v3.2.0/train+public_test/` | Dataset giải nén riêng; ZIP gốc giữ nguyên |
| `data/manifests/split-v1.json` | TRAIN_FIT/VAL_MAIN và 12 smoke IDs thuộc TRAIN_FIT |
| `data/references/train-v1.json` | Gold references từ một job replay train riêng |
| `local_baseline/` | Code setup, chia dữ liệu, model/backend, agent, scorer và lệnh chạy |
| `runs/<run_id>/` | Kết quả từng lần chạy; không ghi đè report cũ |

File `.gitignore` riêng trong LifeMate loại weights, runtime, dataset và runs khỏi Git. Code và tài liệu vẫn hiện trong `git status`; không có thao tác commit/push tự động.

## 2. Mở terminal ở đâu?

```powershell
cd D:\Desktop\workspace\competition\2026\ZAIC\LifeMate
```

Python hiện có trên máy đủ cho code điều phối vì chỉ dùng thư viện chuẩn. Không cần cài PyTorch mới để chạy đường GGUF này. Bộ CUDA portable dùng NVIDIA driver sẵn có. Máy khác cần chuẩn bị driver/GPU phù hợp; script setup hiện nhắm Windows x64/NVIDIA, không phải installer đa nền tảng.

## 3. Các lệnh chuẩn bị, đã thiết kế để không ghi đè dữ liệu khác

```powershell
python -m local_baseline.setup_local
python -m local_baseline.data
python -m local_baseline.references
python -m unittest local_baseline.test_baseline -v
```

Setup tải hai ZIP binary chính thức đã pin release, kiểm digest SHA-256 rồi giải nén riêng. Dataset extraction kiểm đường dẫn, từ chối traversal/symlink và bỏ AppleDouble/DS_Store. File tồn tại chỉ tái sử dụng nếu giống hệt; ZIP gốc được kiểm hash trước/sau extraction.

References là job annotation **train-only** riêng: replay gold bằng official ToolEnv, so các app trong `label.final_state`, lưu full hash từ runtime. Agent run tạo một phiên riêng đúng một lần cho mỗi sample; không dùng reference để thử/reset phiên agent. Khi đã có marker đúng dataset, chạy lại lệnh prepare chỉ dùng lại artifacts.

## 4. Chạy Qwen lần đầu

Kiểm model load và trả lời tiếng Việt:

```powershell
python -u -m local_baseline.run --model qwen --hello-only
```

Chạy đủ 12 mẫu smoke với tool thật:

```powershell
python -u -m local_baseline.run --model qwen --partition smoke
```

Chạy ít mẫu để debug nếu cần:

```powershell
python -u -m local_baseline.run --model qwen --partition smoke --limit 3
```

12 mẫu này là TRAIN_FIT, có zero-call, một call, multi-step, ask và history. Dùng để sửa tích hợp; không dùng score smoke để chọn model thắng.

## 5. Chạy Spark sau Qwen

```powershell
python -u -m local_baseline.run --model spark --partition smoke
```

Chạy lần lượt, không chạy hai model tranh cùng GPU khi đo thời gian. Hai model có native chat template riêng trong GGUF, cùng system instruction, input, tool set, greedy settings, giới hạn context và runtime. Thinking được tắt; raw response vẫn được lưu để kiểm thực tế. Model chỉ load một lần cho mỗi job; server đóng khi job kết thúc hoặc lỗi, không để service thường trực.

## 6. Xem kết quả ở đâu?

Terminal in `RUN_DIR` và `REPORT`. Mỗi run có:

- `SUMMARY.md`: bảng dễ đọc từng sample.
- `metrics.json`: điểm local, state match, tool F1, trạng thái và text latency.
- `scores.jsonl`: điểm từng sample, kể cả failures.
- `submission.jsonl`: đúng ba trường `id`, `tool_calls`, `state_hash`.
- `traces.jsonl`: raw responses, proposals, validation, executed calls/results và lỗi.
- `run_manifest.json`: nguồn weights/hash, split, code, prompt, runtime và các cấu hình.
- `server.log`, `hardware.txt`: chẩn đoán runtime/GPU.

Các artifacts chứa dữ liệu cuộc thi, giữ local/private. Không upload logs/transcripts tự động.

So hai run hoàn tất bằng đường dẫn thật được in ở terminal:

```powershell
python -m local_baseline.compare runs\<run_qwen> runs\<run_spark>
```

Tham số `<run_qwen>` và `<run_spark>` là placeholder, phải thay bằng tên thư mục thật. Muốn lưu báo cáo mới, thêm `--output runs\comparison-moi.md`; lệnh từ chối ghi đè file báo cáo đã tồn tại.

## 7. Khi nào mới chạy validation để chọn model?

Sau khi smoke đã xác nhận pipeline đúng và chốt config. Lệnh có sẵn:

```powershell
python -u -m local_baseline.run --model qwen --partition val-main
python -u -m local_baseline.run --model spark --partition val-main
```

Không tự chạy hai lệnh này đồng thời. Full VAL_MAIN gồm 309 mẫu ở split hiện được tạo; manifest ghi denominator thật và subset nếu có `--limit`. Mọi timeout/overflow/backend failure vẫn xuất prediction/trace và có điểm failure bằng 0, không drop sample. Không sửa prompt dựa từng mẫu validation rồi coi bộ đó là holdout chưa dùng.

Nhánh này là **pilot GGUF Q4_K_M**, không tự gắn nhãn B0/B1/B3 của protocol SFT trong plan. Chưa ASR nên chưa có S0_AUDIO. Sau này cần chốt backend/template/precision cho so sánh trước-sau fine-tune; GGUF pilot và Transformers/LoRA có thể khác precision/runtime nên phải chạy lại baseline phù hợp. Không tự train trước các gate của plan chung.

## 8. Quy tắc của agent đang dùng

- Chỉ train gold-text mode: transcript hiện tại và history lấy từ train; public/private bị từ chối vì chưa có ASR provider.
- Agent nhận view allowlist, không chứa label/final_state/reference hash.
- Mọi installed tools theo catalog đều vào baseline; thêm `sys.ask_missing_info`, không retrieval/shortlist/few-shot.
- History được serialize theo thứ tự như dữ liệu tham khảo đã thực hiện; không replay history. Kết quả lịch sử thiếu được giữ thiếu.
- Function calling qua native chat template, mỗi bước một proposal. `env.validate` kiểm trước execute; tối đa hai repair. Proposal chưa thực thi không đưa vào submission.
- Một ToolEnv/sample; chỉ dùng validate/call/get_tool_calls/get_state_hash trong agent. `get_state` chỉ có trong job reference train riêng.
- Sau executed ask thì dừng. Không giả lập câu trả lời user. Parse lỗi/truncation không coi là finish thành công.
- Tổng context ≤16.384, output reserve 512 token, tối đa 8 executed calls và 30 giây mỗi mẫu gold-text. Call/repair budgets là policy đội, không phải giới hạn BTC.
- Backend tokenize/render trước generate; không tự cắt history/schema. Overflow thành failure rõ ràng. Runtime cũng từ chối generate nếu server báo context quá dài.
- Thời gian text bắt đầu từ chuẩn bị view/ToolEnv đến kết quả; model startup ghi riêng. Chưa chứng nhận hard real-time, chưa đo audio end-to-end/A30.

## 9. Bộ chấm local và giới hạn cần biết

`local-spec-v1` là implementation bám mô tả đã đọc, **không phải official scorer BTC**. Các assumptions được lưu trong run manifest: fuzzy chỉ ở field `x-match:fuzzy`, NFC/lowercase, Unicode word tokens bỏ punctuation, multiset token F1 **>0,6**, non-fuzzy exact arguments/key sets, maximum bipartite matching giữ multiplicity. Tokenizer fuzzy và một số corner cases của BTC chưa được công bố đầy đủ.

Gold không rỗng mà full hash sai hoặc không có matched calls thì score 0; gold rỗng/pred rỗng theo special case. Timeout/execution failure override điểm 0. Tool F1/state accuracy được báo riêng để không che nguyên nhân. Self-replay gold đã được job chuẩn bị kiểm tra, nhưng điều đó không chứng nhận assumptions fuzzy giống server chấm.

Split hiện là grouped 80/20 với seed 20261103: group exact current text/full user dialogue/current audio và near text Jaccard≥0,90 trong cùng gold tool skeleton. Báo phân phối, không tuyên bố stratified hoặc speaker-disjoint. Cần DATA audit thêm near-history/audio và các slice hiếm trước thí nghiệm chính thức.

Backend loopback và bộ chấm này đủ để kiểm đường đi model → tool → observation → prediction. Các bước chưa thực hiện: ASR, training, submission Docker, chứng nhận 30 giây trên A30 và xác định model thắng bằng validation đầy đủ.

Nguồn runtime: [llama.cpp b11800](https://github.com/ggml-org/llama.cpp/releases/tag/b11800), [function calling](https://github.com/ggml-org/llama.cpp/blob/b11800/docs/function-calling.md), [server API](https://github.com/ggml-org/llama.cpp/blob/b11800/tools/server/README.md). Nguồn task: README/catalog/runtime official trong ZIP local và [portal LifeMate](https://challenge.zalo.ai/portal/lifemate).
