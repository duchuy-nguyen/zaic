# Handoff Docker — đúng bản public52

## Tài nguyên cần dùng ngay

- Code inference: folder `inference/`. Module `baseline50` là tên API lịch sử, hiện ghim manifest52.
- ZIP01: PhoWhisper-large merged FP16, Qwen3.5 DPO Q4_K_M, processor/tokenizer, retrieval TRAIN-only, split và references dùng đánh giá. ZIP02: runtime Windows gốc để tham khảo môi trường, không chạy được trong Linux.
- Qwen file: `models/qwen/Qwen3.5-4B-LifeMate-decision-dpo-Q4_K_M.gguf`, SHA256 `aee8106ebe960c26e905443fedb3e8bd2e3a6c1218e63806892f849dbb8e512e`.
- Qwen adapter cuối: `ca1efa57fa770af2cd5d56ce0a85b4e92da493580a75f8e97640eb6167c2b648`; parent SFT adapter: `7d04b7fd5feea6b92c230f4e710df25046a710c5d80ac9d8fe1f4e6dd73c93a5`.
- ASR unchanged so với50.9; adapter Large: `fed80cdce5969ac72b5eae98b4d4fe28bccc9fccae2d39db093e4aa800502662`. Toàn bộ hash file ASR/processor có trong `inference/assets.manifest.json`.
- Logical deployed: Qwen4,205,751,296 + ASR1,543,304,960 =5,749,056,256. Qwen đầy đủ32text blocks, không dùng dropout/pruning ở inference để giảm tham số.

Không nhầm parent GGUF50.9 trong ZIP04 (chỉ dùng TRAIN harvest/reference lineage) với deployed GGUF52 trong ZIP01. Manifest52 sẽ từ chối parent GGUF cũ.

## Dòng xử lý và API

1. Nhận current audio và mọi user audio trong history; ASR16kHz mono, languagevi/tasktranscribe, greedy, max256tokens, EOS bắt buộc. Assistant/tool history giữ nguyên text/tool calls/results được cung cấp.
2. Lập view chỉ có `id`, `latest_text`, `context`, `history`, không label/final state. Bind retrieval qua `actor.bind_package(path_to_private_data)`; bank chỉ từ TRAIN-fit.
3. `actor.infer(view, catalog, env, backend, seconds=remaining, max_calls=8, max_repairs=2)` tự gọi validate/call và quyết định bước kế sau observation thật. Catalog map theo tên tool. Mỗi mẫu dùng một ToolEnv. Không để Ops gọi lại các calls actor đã thực thi.
4. Backend `generate(messages, tools, deadline)` trả `(response_dict, rendered_prompt_tokens)`. Native tool calling; `enable_thinking=false`, greedy, context16384/output512. Runtime phải hỗ trợ `/apply-template`, `/tokenize`, `/v1/chat/completions` đúng template/schema/usage của b11800.
5. JSONL output: ba trường `id`, `tool_calls`, `state_hash` từ actual ToolEnv. Đúng ID/thứ tự SDK yêu cầu. Debug/timing lưu riêng; không đưa labels/references vào actor.

Policy gốc: serial ASR batch2 → chuyển ASR về CPU → Qwen GPU → unload server từng mẫu. Timing30s bao gồm ASR/chuyển thiết bị/startup/tool/hash trong sample. Warmup TRAIN nằm ngoài per-sample; không warmup bằng private test đáp án. Cached-agent VAL không chứng nhận end-to-end thời gian Docker.

## Phần bắt buộc port cho Linux

- Cài runtime llama.cpp b11800 và CUDA dependencies Linux phù hợp máy BTC; không copy .exe/DLL vào Linux. Windows verifier/backend CLI pin hiện không phải bản Linux sẵn chạy.
- `baseline50.io.verify_assets` ghim manifest52 gồm Windows runtime. Tạo một variant manifest/verifier cho Linux, ghi hash runtime mới và giữ đúng model/processor/retrieval hashes. Không xóa kiểm tra để ép chạy và không sửa bản gốc này.
- CLI VAL/public hiện ghim fingerprint dataset/catalog/SDK và634public rows. Private prediction cần loader SDK split/paths BTC tương ứng; không dùng hardcoded634 cho private.
- `LocalServer` dùng tham số launch đúng phiên bản. Kiểm GPU offload, tokenizer/rendered token parity, native argument decoding và server cleanup. Không được ngầm fallback sang external API/Internet.
- Mount dữ liệu runtime read-only theo contract, xuất đúng đường dẫn `/result`; script predict phải có exit code đúng khi thiếu model hoặc lỗi nền tảng.

## Contract bài nộp

Đối chiếu bản Docker Guide/portal BTC đang dùng trước khi nộp. Guide đã được nhóm tham khảo quy định predict.sh đọc `/data`, ghi `/result/submission.jsonl`; có JupyterLab/start_jupyter.sh/predict_notebook.ipynb, notebook tạo `/result/jupyter_submission.jsonl` và ghi timing, kết quả phải khớp prediction thường. Đồng thời phải có training_code, dữ liệu hoặc link cố định và README tái lập. Bộ này cung cấp code/assets/training handoff, không tự tạo image hoặc các script Ops chưa được kiểm thử.

Link tài liệu đã tham khảo: https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf

## Training kèm theo để đóng gói

`training/base50_9` + ZIP03 tái lập SFT Qwen và LoRA ASR. `training/dpo` + ZIP04 cung cấp preference/replay, approved review63, reference cache, trainer và checkpoint63. Xem TRAINING_REPRODUCTION.md để chạy từ đầu hoặc merge adapter lịch sử mà không train lại. ZIP05 chứa traces/metrics/submission để đối chiếu, không phải input model.

Không cần chạy lại training để dựng Docker52. Tách tài nguyên training khỏi actor input khi inference; tất cả model cần thiết cho offline inference phải có sẵn trong image.

## Trạng thái kiểm chứng

Nguồn52 đã chạy trên Windows RTX4060,634public0timeout. Package này được kiểm theo VALIDATION.json. Chưa build Docker, chưa chạy Docker trên GPU BTC và chưa xác nhận variant Linux/private loader. Không coi CPU/package audit hoặc Test Solution của50.9 là chứng nhận Docker52.

`docs/PARENT50_9_DOCKER_HANDOFF.md` chỉ là tài liệu lịch sử bổ sung về API. Model/manifest/score hiện tại lấy từ file này và RELEASE_IDENTITY.json.
