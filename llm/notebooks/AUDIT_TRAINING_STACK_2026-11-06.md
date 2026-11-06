# Audit notebook training stack — 2026-11-06

Phạm vi: bootstrap, train ASR/student, teacher rollout, inference/evaluation, checkpoint/resume, artifact reuse và export trong `LifeMate_OneClick.ipynb`. Đây chưa phải chứng nhận full model trên Kaggle T4. Không đọc initial environment files, không tải pretrained weights để test local, không train trên public/private test.

## Kết luận và sửa lỗi

| Mức | Finding | Sửa/giới hạn |
| --- | --- | --- |
| P0 | Student 8K OOM trong SDPA. QLoRA/checkpointing không ngăn backend dựng score matrix toàn chuỗi. | Backend `lifemate_bounded_128`: full causal attention theo khối query, giữ mọi keys, checkpoint từng khối cho backward. Không sliding window/truncate. Mask dựng từng khối trên device của layer, hỗ trợ padding/cache offsets. |
| P0 | Selected-position loss vẫn có thể dựng toàn bộ target × vocabulary logits cho completion dài. | Hidden forward một lần, lm_head + exact summed CE theo 32 target positions/khối, recompute backward rồi chia tổng số target hợp lệ. |
| P1 | PEFT upcast frozen embedding/lm_head của vocabulary 248K lên FP32. | Khôi phục hai module frozen về FP16 đã dùng lúc load, giữ norms/adapters FP32. Không đổi parameter count. Với embedding/head 248320 × 2560 untied, chênh storage lý thuyết khoảng 2.37 GiB; xác minh freed bytes tại runtime. |
| P1 | Smoke diễn ra sau teacher và base eval; lỗi xuất hiện quá muộn. | `stack_check` và `student_precheck` lên trước ASR/teacher; actual-data smoke chạy trước student base eval. Gold-fit context coverage cũng kiểm tra sớm. |
| P1 | Smoke cũ có thể đếm optimizer step dù AMP skip do overflow. | Smoke kiểm tra gradient hữu hạn và có gradient; init scale 128 dùng thống nhất với Trainer. Test GPU phát hiện scale mặc định 65536 overflow bước đầu trên tiny NF4 model. |
| P1 | Mất thời gian sinh lại teacher sau khi sửa student. | `REUSE_TEACHER_FROM`: kiểm tra source bundle/config/data/split/references/stage hashes rồi re-score mọi record; chỉ nhập completed trajectories với acceptance không đổi. Không replay ToolEnv. ASR reuse có receipt riêng. |
| P1 | Chat EOS khác checkpoint EOS. | Giữ fix v5: truyền hợp checkpoint/chat EOS trực tiếp vào generate cho teacher/student. |
| P1 | venv/ensurepip/sitecustomize trên Python 3.13. | Giữ isolated venv, `--without-pip`, PyPA bootstrap pip, worker `-I`, wrapt pin, xác minh CUDA trước model loading. |
| P2 | Model 27B bị đổi sang 9B trong các lượt xử lý trước. | Bản sửa student này không đổi model nữa. Khi import v5, corpus teacher được ghi đúng model/revision đã sinh nó (9B), không gọi đó là dữ liệu từ 3.8. 3.8-27B/T4 vẫn chưa được chứng nhận; cần experiment riêng. |

## Evidence đã chạy

- 40 offline regression tests: parser, schema permission, ToolEnv log source of truth, EOS, split, archive, budget, reuse, numerical attention/gradient/CE equality và memory tile bounds trong cả forward/backward.
- Tiny random Qwen hybrid model với Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0, bitsandbytes 0.50.2: NF4 load, real Trainer/AMP optimizer steps, checkpoint ở step 1, controlled pause, resume step 2, save/reload adapter.
- Resume adapter so với uninterrupted run: maximum difference **0.0**.
- Native vs bounded attention trên tiny full model: max logits difference **2.68e-7**, max parameter-gradient difference **4.66e-9**; cached greedy generation khớp.
- Long probe: **8192 tokens**, 16 Q heads / 4 KV heads / head_dim 256, tiny backbone, 2 forward/backward/optimizer steps: peak allocated **1.26 GiB**, peak reserved **1.51 GiB** sau frozen-IO compaction. Đây không phải footprint full 4B.
- Máy local: RTX 2050 4 GB, compute capability 8.6, Windows, Python 3.14.3, torch 2.9.0+cu128. Khác Kaggle T4 sm75/Python 3.13/torch 2.9.1+cu126; không dùng kết quả local để chứng nhận T4.
- Đã extract chính payload notebook và chạy `python -I .../worker.py stack_check ...` qua main/stage marker thật; không chỉ import source trực tiếp.

Evidence local (ignored, fixtures only): `runs/oneclick_compat_results/check_20261106_d/report.json`, `runs/oneclick_compat_results/final_cli/stack_check.json`. Không có BTC data trong các report này.

## Bước xác minh còn thiếu

1. Trên T4 của team, stack check phải pass trước các job dài.
2. Full Qwen3.5-4B resource probe ở **context đã cấu hình 8192**, có optimizer state, phải pass. Không tự giảm context hay hạ coverage gate.
3. Actual-data smoke, train, text/audio eval, export phải hoàn tất; xem peak/runtime/loss/score trên đúng IDs.
4. Quality, latency 30 giây trên hardware BTC, Docker offline, parameter accounting và determinism cuối cùng vẫn cần kiểm chứng riêng.

Không cam kết hết mọi lỗi trên target GPU khi chưa hoàn tất các bước trên. Bản v6 đưa kiểm tra target GPU lên đầu, không yêu cầu người dùng chạy teacher/ASR hàng chục phút trước khi biết student có train được không.

Nguồn kỹ thuật: [PyTorch SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html), PyPI Transformers 5.20.2 wheel SHA256 `7ae75dbc1fb3922aff1b6c97b9e5479d6cf26fd86a8866bfb71324190479f807` đã đối chiếu source attention/masking/Trainer thực tế.
