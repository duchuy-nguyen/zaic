# DATA → OPS — bàn giao 09/11/2026

Trả lời `ops/docs/ops/HANDOFF_REVIEW_2026-11-05.md` (mục "Bàn giao tiếp theo" 1), checklist "DATA bàn giao"
trong `ops/docs/ops/CHECKLIST.md`, và các mục còn mở của release `synthetic_phase1_v21_text_551`.

## 1. Checklist "DATA bàn giao"

| Mục | Bàn giao | Ở đâu |
|---|---|---|
| Archive SHA256, split name/manifest SHA256, ID list, catalog SHA256 | `split-v1`; hash split/catalog/runtime/archive; ID list theo định dạng `PREFLIGHT.md` | git: `data/releases/ops_train_smoke_v1/{manifest.json,provenance.data.json}`, `data/manifests/expected_ids_{fit,val}.jsonl` · Drive: bundle (`ids/` có thêm `smoke`, `public_test`) |
| Source ToolEnv nguyên bản đúng split, runtime revision/hash, API đã xác nhận | runtime + catalog **train** nguyên bản trong bundle; hash runtime/catalog **public_test** trong manifest; API ghi ở `toolenv_api_confirmed` | Drive: `ops_train_smoke_v1` |
| Input sạch label/state; history + transcript source | `inputs/inputs_smoke.jsonl` (không có label/state; `transcript_source = gold_audio_text_train_only`) | Drive: bundle; VAL/FIT: `eval_v1/inputs_*.jsonl` |
| Evaluator/scorer version, references train-only tách inference | `data-eval-v1` (`data/src/lifemate/eval/scorer.py`); references `id, tool_calls, state_hash` tách thư mục | git: code · Drive: `references_train_only/`, `eval_v1/references_*.jsonl` |

### Bộ smoke `ops_train_smoke_v1`
- 12 mẫu TRAIN_FIT phủ: write_only, lookup_then_write, read_only, ask (entity/scalar), empty; history 0 / 1–4 / 5+.
- Root layout đúng `ToolEnvFactory(ToolEnv, root=<bundle>, split="train")`; đưa `<bundle>/train` lên PYTHONPATH (một runtime / process).
- `state.json` dựng qua `ToolEnv("train").get_state()` + `_meta` từ context, không copy file gốc. Đã kiểm trên 1545/1545 mẫu train rằng cách này cho state_hash và replay giống phiên gốc.
- Đã chạy ở DATA:
  - `verify_bundle.py`: BUNDLE OK.
  - Chạy `ToolEnvFactory` + `ToolEnvAdapter` của OPS bằng gold calls: hash cuối khớp 12/12.
  - Toàn trình (input sạch → adapter OPS → `session_output()` → evaluator DATA): **100.0**.

### Evaluator `data-eval-v1`
- Cài đặt đúng công thức của đề: state hash, call matching (fuzzy token-F1 > 0.6), P/R/F1, gold rỗng.
- Các điểm đề không nói rõ được ghi thành giả định A1–A4 trong docstring; tokenizer chọn bằng `--tokenizer whitespace|word`.
- Kiểm tra:
  - Nộp đúng gold được 100.0; nộp toàn rỗng được 7.1429 (= tỉ lệ gold rỗng của VAL).
  - **1232/1232** dự đoán nhiễu có hash replay thật cho điểm trùng `llm/local_baseline/scoring.py` (local-spec-v1, chế độ `word`).
  - Hai tokenizer cho điểm khác nhau ở 0/1232 ca.
- References VAL: 308 mẫu (action 242, ask 44, empty 22); FIT: 1237 mẫu. `state_hash` gold = replay gold trên ToolEnv train.

## 2. Mục còn mở của release 551

| Mục OPS ghi chưa kiểm | DATA |
|---|---|
| `synth_v21_t2_00016` | **Sai nhãn, đã xác nhận** ("Mai Anh nhà chị" chỉ người nhà). Xem `ADJUDICATION.md`. Bản 551 là bản v2.1 dùng train text. |
| Train-state provenance | C8 trên nguồn 552 (`validation_v21.json`, 0 lỗi): t3/t4 cùng hash state với mẫu train gốc; t2 = state gốc + đúng 1 thực thể. Bản 551 là tập con của 552. |
| VAL text leakage | C5 trên nguồn 552 (0 lỗi): mọi `source_id` thuộc FIT; không transcript nào trùng câu (kể cả history) của VAL split-v1. |
| Audio T1/T2 (100 mẫu) | C10: `audio` và `audio_text` **giống hệt** cặp train gốc của BTC. Không có audio mới, nên mức sẵn sàng cho ASR bằng đúng cặp train gốc. |
| Semantic adjudication độc lập | Còn mở: `audit_phase1.csv` cần người duyệt. |

## 3. CI DATA (`scripts/ci/check_data.py`)

CI cũ chỉ kiểm `data/synthetic/synthetic_phase1.jsonl`, là bản **v1 488 mẫu đã lỗi thời** (có lỗi `_syn`). CI mới:
- Kiểm split; với **mọi release** trong `data/releases/`: sha256 file == `dataset_sha256`, số dòng, số mẫu theo kỹ thuật, ID trùng, ID đã loại, tool có trong catalog, nguồn thuộc FIT. Kiểm thêm manifest bundle OPS.
- Chặn data thô (`*.wav`, `state.json`, `*.tar/*.gz/*.zip`). Với `.jsonl` trong `data/releases/` chỉ cảnh báo, không chặn.
- Đã chạy thử:
  - Trên `main` hiện tại: pass (kèm cảnh báo file v1).
  - Sửa 1 dòng trong release 551: báo lỗi lệch hash.
  - Thêm `state.json`: báo lỗi.
- Commit này gỡ `data/synthetic/synthetic_phase1.jsonl` và `synthetic_phase1_meta.json` (v1). **Giữ** `data/synthetic/zaic_tools` và `synthetic_tools_catalog.json` vì test của OPS dùng.

## 4. Cần nhóm thống nhất
- Nhóm trưởng: "không push data lên git"; OPS đang commit `synthetic_phase1.jsonl` của release lên `main`. CI mới không chặn trường hợp này. Cần chốt một quy định.
- Baseline50 của LLM đóng băng FIT1236/VAL309, còn DATA split-v1 là 1237/308. So sánh kết quả giữa hai bên phải dùng cùng một split.
