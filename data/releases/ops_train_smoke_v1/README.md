# ops_train_smoke_v1 — bộ train smoke cho OPS

12 mẫu **TRAIN_FIT** (split-v1) kèm environment, catalog và runtime train nguyên bản, đúng root layout của
`ToolEnvFactory(ToolEnv, root=<bundle>, split="train")`.

| id | loại | số call | app |
|---|---|---:|---|
| `train_0168` | write_only / h0 | 1 | reminders |
| `train_0028` | lookup_then_write / h0 | 2 | alarm, calendar |
| `train_0677` | lookup_then_write / h1-4 | 3 | contacts, ewallet, notes |
| `train_1721` | lookup_then_write / h5+ | 3 | banking |
| `train_0960` | read_only / h0 | 1 | news |
| `train_0605` | ask_entity / h0 | 2 | chat_ott, sys |
| `train_1437` | ask_entity / h1-4 | 1 | sys |
| `train_0683` | ask_scalar / h0 | 1 | sys |
| `train_1634` | ask_scalar / h5+ | 2 | loyalty, sys |
| `train_1167` | empty / h0 | 0 | — |
| `train_0655` | empty / h1-4 | 0 | — |
| `train_1664` | write_only / h5+ | 1 | tasks |

## Dùng
```bash
cd ops_train_smoke_v1
python verify_bundle.py                       # phải in BUNDLE OK
PYTHONPATH=$PWD/train:<repo>/ops/src:<repo>/llm python ...   # một runtime / process
```
```python
from zaic_tools import ToolEnv                 # từ ops_train_smoke_v1/train
from lifemate.adapters.toolenv import ToolEnvFactory
factory = ToolEnvFactory(ToolEnv, root=Path("ops_train_smoke_v1"), split="train")
```
## Nội dung theo checklist "DATA bàn giao" (ops/docs/ops/CHECKLIST.md)
| Mục OPS | File |
|---|---|
| Archive/split/ID list/catalog SHA256 | `manifest.json`, `provenance.data.json`, `ids/expected_ids_{smoke,fit,val,public_test}.jsonl` |
| Source ToolEnv đúng split, runtime revision/hash, API | `train/zaic_tools`, `manifest.json` (`runtime_*`, `toolenv_api_confirmed`, `public_test`) |
| Input sạch label/state; history + transcript source | `inputs/inputs_smoke.jsonl` (`transcript_source` = gold_audio_text_train_only) |
| Evaluator version + references train-only tách inference | `references_train_only/references_smoke.jsonl`; scorer `data/src/lifemate/eval/scorer.py` (data-eval-v1) |

`references_train_only/references_smoke.jsonl`: gold `tool_calls`, `state_hash` sau khi replay gold, `initial_state_hash`
-> so với `session_output()` khi chạy đúng gold calls; chấm bằng
`python -m src.lifemate.eval.scorer --submission sub.jsonl --references references_smoke.jsonl --catalog train/train_tools_catalog.json`.

## API ToolEnv đã xác nhận (zaic_tools/README.md + executor.Session, đã chạy thử)
- `ToolEnv(split, sample_id, root=<root>)`; `env.validate(tool, args)` -> `{ok, error}` (không chạy);
  `env.call(tool, args)` -> dict có `ok` (mọi call, cả lỗi, đều được log);
  `env.get_tool_calls()`, `env.get_state_hash()` = hai trường của dòng nộp.
- `env.get_state()` chỉ dùng debug/so label TRAIN; cấm trong đường inference.

## Hash (điền provenance)
- split_name `split-v1`, split_manifest_sha256 `8c3a5f395fc95de1813483b42768bd7d29a9df691d67f5899708e4c509ce6a5e`
- catalog_sha256 `a994984bb580d7e0d60166641858901d428549682ba3614dbf9269037a33cce0` (schema 2.7.0)
- runtime_sha256 `8bb0a170da15f458b606d41665c82970389155b25c731708db0f6bcdf81dc412`
- archive_sha256 `57d371518996ec70d3af86402b7eba61208093c82955b9807a7ab39604b47320`
- public_test: `{"n_ids": 634, "catalog_sha256": "c8f46c278a7495d36905ac28fc5b85b67aad22bc2ac0b29ba9ac6455ad9daf5d", "catalog_schema_version": "2.7.0", "runtime_revision": "lifemate-v3.2.0/public_test/zaic_tools (catalog schema 2.7.0)", "runtime_sha256": "8c5e229dd7eb144531b105824f599d4d842285c36ac4be5be5a14619ba837053", "n_tools": 122, "note": "runtime/env public_test lấy từ archive BTC (đúng archive_sha256); bundle không chứa public_test"}`
Chi tiết: `manifest.json`, `provenance.data.json`.

## Lưu ý
- `inputs/` không chứa label/state; `references_train_only/` chỉ cho evaluator. Public/private phải dùng transcript ASR.
- `state.json` dựng qua `get_state()` + `_meta` từ context (không copy file gốc); state_hash trùng phiên ToolEnv gốc.
- Đây là data của BTC: để trên Drive, không commit lên git.
