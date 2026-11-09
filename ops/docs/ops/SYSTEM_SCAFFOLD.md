# Khung inference LifeMate

Baseline mới `llm/baseline_50_2` có [bridge riêng](BASELINE50_BRIDGE.md) để giữ
nguyên actor/prompt/retrieval và ASR batch. Không chỉ thay transport của scaffold
rồi coi là cùng baseline. CLI dưới đây hỗ trợ factory context manager và smoke cap.

Khung nằm trong `ops/src/lifemate/system/`, dùng lại Runner và adapters hiện có. Chạy demo không cần GPU, network, weights hoặc data BTC. Đây là khung có thể tích hợp, chưa phải solution đã chứng nhận trên Kaggle/A30.

## Chạy ngay bằng fixture

Từ repository root, chọn thư mục output **mới**, giữ private:

```powershell
python ops/scripts/ops/run_system.py --factory lifemate.dev.system_demo:build_system --input ops/configs/ops/system_demo_input.jsonl --output-dir ops/runs/system_demo_001
```

Linux/Kaggle chạy cùng lệnh. Entry point thêm `ops/src` và `llm` vào import path; không cần `pip install` package Ops. Giữ checkout đủ `ops/`, `llm/` khi dùng native adapter. Demo hoàn toàn scripted, dùng `FixtureSession`; hash trong output là hash fixture, **không phải hash BTC**. Không upload output demo lên portal.

Kết quả kỳ vọng: `status=completed`, `completed=2`, `failures=0`. Mỗi sample có lookup → append → STOP, một session riêng. Script từ chối ghi đè output directory.

```text
ops/runs/system_demo_001/
  run_manifest.json       # IDs, runtime/config/catalog hashes, provenance, completion/failure counts
  traces.jsonl            # retrieval, model, repair, validation, execution, stop
  diagnostics.jsonl       # status, total latency, call counts; không phải official score
  submission.jsonl        # calls/hash từ cùng session; không reconstruct từ proposals
  submission_check.json   # coverage/format checks, không xác minh quality/hash authenticity
```

`completed_with_failures` vẫn xuất các record có thể xuất được với hash thật; CLI trả exit 1 để báo lỗi. Nếu không tạo được environment hoặc không lấy được hash, batch ghi manifest `failed` rồi dừng, không bịa record hoặc mở session thay thế. Không tự resume/replay partial batch; giữ log cũ để kiểm tra side effects trước.

## Luồng và điểm mở rộng

```text
Prepared text/audio JSONL
  → one environment/session (ID only)
  → prepare_sample: ASR khi cần + allowlist context/history
  → Runner với thời gian còn lại sau ASR
  → NativeLLMBackend + PromptAugmenter
      → permitted schemas → optional ToolSelector
      → per-sample grounding evidence + optional ExampleBank
      → injected transport.generate()
  → GuardedEnvironment: permissions + evidence binding
  → ToolEnvAdapter.validate() → execute() → observation
  → repeat / ASK / STOP
  → original environment.session_output() → submission + diagnostics
```

| Module | Đã có | Giới hạn / việc tiếp theo |
| --- | --- | --- |
| `context.py` | Prepared text/audio boundary; ASR protocol nhận remaining time; không dùng `audio_text` gold | Cần ASR adapter thực và kiểm tra đường audio/metadata/cache trong adapter |
| `retrieval.py` / `ToolSelector` | CPU lexical word + character cosine, deterministic tie-break, dependency closure, giữ ASK, repair/zero-evidence fallback full | Không phải BM25/dense RAG; mặc định top_k=0 tắt pruning. Cần đo tool recall và downstream score |
| `retrieval.py` / `ExampleBank` | Procedural demonstrations, kiểm tra ID thuộc fit manifest được caller cung cấp, hash corpus/split | Không tự xác minh split do caller khai báo; không tự delexicalize; không nhận executable calls/argument IDs |
| `grounding.py` | Exact typed bindings từ source paths đã cấu hình; nguồn context hoặc observation thành công; explicit write invalidation | Chưa phải fuzzy resolver/NER hay temporal normalizer. Binding phải kiểm tra theo schema thật |
| `prompt.py` | Recompute retrieval/evidence mỗi bước; trace names/counts/example provenance; giữ raw history/results | Chưa compact history hoặc suy luận task ledger; không cắt context âm thầm |
| `pipeline.py` | Một environment, cùng deadline ASR/agent/export, export từ session thật, manifest cấu hình | Deadline là cooperative; adapter blocking không bị hard-kill. Không claim hard real-time |
| `cli.py` | Prepared JSONL batch, coverage check, private traces, failure manifest, không overwrite | Chưa load official audio manifest/catalog/split runtime tự động; không chạy private test mặc định |
| Native transport hook | Tái sử dụng `generate(messages, tools, deadline, max_tokens=...)` | Constrained decoding/grammar phải tích hợp trong transport theo native format và EOS; scaffold chưa triển khai grammar |

Không đổi teacher/student/ASR model mặc định của notebook training. Notebook hiện tại **chưa embed khung inference này**; demo hoặc factory dùng source checkout. Không suy ra notebook evaluator và Ops đã parity chỉ vì dùng cùng model: vẫn phải so prompt/parser/STOP/budgets với contract tests và actual model runs.

## Input contract

CLI nhận input đã chuẩn bị, không nhận nguyên dòng train có label. Một dòng ví dụ:

```json
{"id":"sample_id_from_input","latest_text":"yêu cầu đã ASR","context":{"current_time":"2026-11-07T10:00:00+07:00","timezone":"Asia/Ho_Chi_Minh","installed_apps":{"app_from_catalog":{"tools":["app_from_catalog.tool_from_catalog"]}}},"history":[]}
```

Tên app/tool trên chỉ mô tả contract, không phải tool để execute. Factory production nạp đúng catalog/split của BTC. Với audio, thay `latest_text` bằng `latest_audio` chứa reference cho ASR adapter. User turn trong history có `text` hoặc `audio`. Nếu truyền cả text và audio, **text được ưu tiên**, không gọi ASR; đây là chế độ prepared input, caller phải ghi rõ text-oracle hay actual ASR trong provenance.

History được đưa vào prompt như dữ liệu đã áp dụng, không replay. Chỉ các field context `current_time`, `timezone`, `location`, `current_location`, `user_profile`, `installed_apps` được giữ. Nếu kit có context field cần thiết khác, thêm có kiểm chứng vào boundary và test. Không trộn label/ref hash/initial state vào profile hoặc history.

## Nối model/ASR/ToolEnv thật

Tạo một module trusted, ví dụ `team_factory.py` tại repo root, với `build_system()` trả `LifeMateSystem`. Factory tự khởi tạo model **một lần** và chịu trách nhiệm seed, pinned revision/dependencies, offline cache, package/resource checks.

```python
from pathlib import Path
from lifemate.adapters.toolenv import ToolEnvFactory
from lifemate.system import LifeMateSystem, ToolSelector

def build_system():
    # Team implementations: not provided/guessed by this scaffold.
    catalog = load_verified_catalog()   # dict keyed by exact tool name
    ToolEnv = import_verified_split_runtime()  # one pinned split package/process
    transport = load_pinned_native_transport() # generate(...) interface below
    asr = load_pinned_asr_adapter()             # transcribe(...) interface below
    return LifeMateSystem(
        catalog=catalog,
        environment_factory=ToolEnvFactory(ToolEnv, root=Path('/data'), split='public_test'),
        transport=transport, transcriber=asr,
        selector=ToolSelector(top_k=0), # full permitted catalog control
        provenance={
            'mode': 'audio', 'model_id': 'ACTUAL_ID', 'model_revision': 'ACTUAL_REVISION',
            'asr_revision': 'ACTUAL_REVISION', 'split_hash': 'ACTUAL_SPLIT_HASH',
            'data_version': 'ACTUAL_VERSION', 'seed': 0,
        },
    )
```

Các hàm `load_*`/`import_*` trên là điểm tích hợp, chưa tồn tại; không chạy nguyên ví dụ như code hoàn chỉnh. `ToolEnvFactory` chỉ gọi constructor runtime; solution không mở state files. Cần checkout/source data kit từ đúng split và không thay bằng runtime synthetic khi chạy official.

Transport trả `(native_response_dict, usage)` từ `generate(messages, tools, deadline, *, max_tokens)`. Response hiện được parse theo native OpenAI-compatible `choices/message/tool_calls`, một call mỗi lượt. Transport Transformers cần render đúng chat template, parse native output đúng format và trả finish reason chính xác; không giả STOP khi output malformed/truncated. Không yêu cầu transport gọi API/network; local HF adapter có thể implement interface này. Nếu dùng server loopback, kiểm tra cách BTC cho phép process/network offline trước khi chọn.

ASR implement `transcribe(audio_reference, *, remaining_seconds) -> str`. Kiểm tra path trong data/audio root, không đọc environment directories; kiểm tra WAV metadata, xử lý deadline và history audio. Không dùng `audio_text` gold làm fallback khi ASR lỗi. Model/processor/cache pin phải ghi provenance. Không cần ASR adapter khi input toàn prepared text.

Sau khi factory được triển khai:

```powershell
python ops/scripts/ops/run_system.py --factory team_factory:build_system --input prepared_inference.jsonl --output-dir ops/runs/inference_001
```

Entrypoint thêm repository root, `ops/src` và `llm` vào import path, nên `team_factory.py` ở repo root được import trực tiếp. Với factory trong package khác, cài package đó vào môi trường Python của team.

## Bật retrieval / grounding / examples

- Tool retrieval: `ToolSelector(top_k=8, dependencies={...})`; dependency names phải lấy từ catalog thực và chỉ được giữ nếu permitted. Không coi k=8 là giá trị tối ưu; so với top_k=0 trên fixed dev. Fallback full khi repair hoặc query không có lexical evidence không đảm bảo recall mọi trường hợp.
- Grounding: khai báo `GroundingRule(tool, argument, sources, invalidate_on=...)`. Source `FactSource(tool_name, ('list_field','*','id_field'))` đọc observation; `FactSource(None, ('context','user_profile','known_id'))` đọc prepared context. Paths là tuple keys, `*` duyệt list/dict; không suy đoán `x-ref` string của catalog.
- Guard chỉ kiểm tra các argument đã có binding; các field khác vẫn dựa vào official `validate`. Thiếu required param do official validation quyết định. Không đoán ID, không tự thêm lookup/ASK hoặc sửa tiền/ngày giờ để vượt guard. Evidence chứng minh ID có nguồn, **không chứng minh đó là entity user muốn chọn**.
- `invalidate_on` khai báo write tools làm cũ ID evidence; sau đó cần nguồn mới. Nếu không cấu hình, scaffold không tự suy ra cache invalidation. Không dùng fact store xuyên sample.
- Examples: `ExampleBank(records, fit_ids=frozen_fit_ids, split_hash=..., top_k=1)`. Mỗi record chỉ gồm `id`, `utterance`, `guidance`, `tools`. Giữ procedural guidance delexicalized, không chèn ID/money/address của sample khác. Manifest caller phải thật sự loại dev/public/private; kiểm tra membership của scaffold không thay thế leakage audit.

## Tests và các gate bàn giao

Chạy từ repo root, source-only, không tải model:

```powershell
python -X utf8 -c "import sys,unittest; sys.path[:0]=['ops/src','llm']; result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover('ops/tests/ops')); sys.exit(not result.wasSuccessful())"
```

Tests bao phủ fixture end-to-end, runtime synthetic đã commit qua adapter, grounding/repair không execute sai proposal, installed tools, calls thất bại/unknown vẫn được giữ, ASK/STOP, sample isolation, audio history, ASR failure/deadline, gold stripping, retrieval dependencies/fallback, example-fit membership, batch coverage và không overwrite.

Trước production: LLM owner nối actual transport và chứng minh format/EOS/context bounds; Speech owner nối ASR và entity metrics; Data owner freeze split/catalog/index; Ops chạy actual model + official ToolEnv integration, determinism/offline/Docker; Research owner đo từng ablation và tổng audio score/latency trên fixed IDs. Chưa có official evaluator, score mới, GPU performance claim hoặc Docker inference certification trong scaffold này.
