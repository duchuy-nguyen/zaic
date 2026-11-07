# LifeMate — Ops nền tảng (demo mock)

Đã có [khung hệ thống inference](docs/ops/SYSTEM_SCAFFOLD.md) gồm prepared
text/audio boundary, optional lexical tool/example retrieval, grounding guard,
agent harness và export từ cùng session. Chạy fixture không cần GPU bằng
`python ops/scripts/ops/run_system.py --factory lifemate.dev.system_demo:build_system --input ops/configs/ops/system_demo_input.jsonl --output-dir ops/runs/system_demo_001`
từ repository root. Actual ASR/model/official split vẫn cần adapter và đo kiểm.

Đã thêm adapter nối source ToolEnv từ DATA và prompt/parser từ LLM;
kiểm tra tích hợp bằng fixture đã pass. Xem [rà soát bàn giao 05/11](docs/ops/HANDOFF_REVIEW_2026-11-05.md).
Chưa chạy model hoặc dữ liệu chính thức qua Ops.

Demo điều phối chạy offline trên CPU, chỉ dùng thư viện chuẩn Python 3.11.
Demo mock không cần model, GPU hay dữ liệu chính thức. Đây chưa phải B0A hoặc submission:
backend, environment và tên tool `demo.*` đều là giả lập phát triển.

## Setup và chạy

Bộ tests adapter cần checkout đầy đủ repo với `llm/` và `data/` nằm ngang hàng
`ops/`. Demo mock riêng vẫn chỉ cần source Ops.

Chạy từ thư mục chứa README này. Dùng một Python 3.11 đã có sẵn để tạo môi trường
riêng; không thay Python hệ thống, không cần `pip install` để chạy demo/tests:

```sh
python3.11 -m venv .venv
.venv/bin/python scripts/ops/demo.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/ops -v
```

Windows: dùng `.venv\Scripts\python.exe`; đặt `PYTHONPATH=src` bằng cú pháp shell
của bạn khi chạy tests. Python mục tiêu là 3.11; không suy ra compatibility của
stack model/GPU từ demo stdlib này.

Chọn config và thư mục run mới nếu cần:

```sh
.venv/bin/python scripts/ops/demo.py --config configs/ops/runtime.example.json --output-dir runs/mock_demo_001
```

Không ghi đè run cũ. Mặc định tạo `runs/mock_<UTC timestamp>/`. Budget là cấu hình
phát triển, chưa freeze B0–B3. Config có deadline 30 giây/sample, 8 executed calls,
16 model steps, 2 repair attempts/sample và 512 output tokens/step; mock không thực sự tokenize/generate.

## Các kịch bản và kết quả mong đợi

| Sample | Kịch bản | Stop reason | Calls đã chạy |
|---|---|---|---:|
| `lookup_update` | Lookup, lấy ID từ observation rồi update | `finish` | 2 |
| `ask_stop` | Ask, chặn lượt model phía sau | `ask` | 1 |
| `finish_no_action` | Finish không action | `finish` | 0 |
| `invalid_schema` | Update thiếu ID, hết repair budget | `repair_budget` | 0 |
| `model_error` | Backend báo lỗi | `model_error` | 0 |
| `deadline` | Backend trả call đúng lúc hết hạn | `deadline` | 0 |
| `parse_repair` | Parse lỗi, LLM repair rồi trả calls rỗng hợp lệ | `finish` | 0 |
| `not_done` | `ok=true, done=false` rồi model finish | `finish` | 1 |

Kịch bản deadline dùng fake clock, không sleep; các kịch bản còn lại dùng
`time.monotonic`. Các số timing này không phải benchmark model/hardware.

## Đọc logs

- `run_manifest.json`: mode mock, config đầy đủ, UTC timestamp, Python/OS,
  Git revision/dirty nếu workspace có Git, fingerprint source/scripts/pyproject,
  provenance và danh sách metadata còn thiếu.
- `tool_traces.jsonl`: sự kiện `model`, `validation`, `execution`, `repair`, `stop`.
  Mỗi dòng có `sample_id`, `step`, `proposal`, `validation`, `executed`,
  `observation`, `latency_seconds`, `stop_reason`; field không áp dụng là null.
  Model event giữ proposal; repair event ghi loại lỗi, budget và có lên lịch repair hay không.
  Chỉ execution event có `executed=true` xác nhận call đã chạy.
- `mock_results.jsonl`: một result cho mỗi sample, stop reason, danh sách call
  đã chạy và observation, calls có trạng thái chưa rõ, last observation,
  elapsed time và `failure` của điều phối.

`finish` chỉ là model ngừng đề xuất, `ask` là chờ người dùng; cả hai không xác
nhận task thành công. `failure=false` nghĩa là không gặp lỗi điều phối;
`done=false` vẫn được giữ nguyên kể cả khi model finish. Không có field
`task_success`, hash state giả hoặc file `submission.jsonl`.

Không đưa secrets vào model input, proposals hoặc observations. Writer che các
key thông dụng như password, API key, token và Bearer credential; runner không
log exception text, raw model output hay input/history. Bộ lọc key không nhận
biết mọi secret ẩn trong văn bản tùy ý. Logs chỉ dành cho local/private và bị
`.gitignore` loại trừ; adapter thật phải có chính sách log được review.

## Chính sách runner

Mỗi `Runner.run` gọi environment factory đúng một lần cho sample đó; factory
phải tạo instance mới, không trả singleton. Backend nhận observations của các
lượt đã chạy và thời gian còn lại, không được truy cập environment. History
chỉ là input cho model, không được runner replay.

Mỗi response đã parse thành công có tối đa **một** call. Runner thực hiện
`model → validate → execute → observation → model`, không chạy song song.
Response có nhiều calls bị từ chối toàn bộ và gửi protocol error về LLM repair;
không tự lấy call đầu tiên. Chỉ `Calls(())` (adapter ánh xạ từ JSON calls `[]`
đã parse và kiểm contract thành công) kết thúc bình thường. Raw `[]`, response
sai kiểu, parse error hoặc timeout không được coi là finish.

Lỗi parse và schema gửi qua `ModelRequest.repair` (`RepairFeedback`) ở lượt kế
tiếp; prompt/parser và sửa arguments thuộc LLM. Ops không tự sửa nội dung.
`max_repairs=2` cho phép tối đa 2 lượt repair dùng chung toàn sample, không reset
sau call thành công; có thể đặt 0 để tắt. Hết budget ghi failure `repair_budget`.
`repairs_used` đếm lượt repair đã lên lịch; vẫn chịu `max_steps` và deadline,
nên một lượt đã lên lịch có thể chưa kịp gọi nếu hết hạn. Proposal bị reject
không vào `executed_calls`. Unknown execution không repair/retry.

Observation `ok=false` hoặc `ok=true, done=false` được giữ nguyên và đưa về model;
không tự retry tool, không suy task success. Model tự quyết định lượt tiếp theo.
Calls đã execute luôn được giữ, kể cả hành động trả lỗi. `terminal` hiện chỉ là
metadata observation, không thay chính sách chuyển tool error về model.
Khi model đề xuất `sys.ask_missing_info`, runner validate/execute rồi dừng;
mock dùng `demo.ask` qua classification `Validation.is_ask`, không giả lập API
ToolEnv chính thức. Ask chưa execute không kết thúc sample.

Deadline bắt đầu trước environment creation, kiểm trước/sau backend, sau
validation và ngay trước execute; cũng kiểm sau execute. Call đã bắt đầu trước
hạn có thể hoàn tất trễ và vẫn phải được ghi lại. Đây là deadline **hợp tác**,
không ngắt được hàm đang block. Trace synchronous trước execute tiêu tốn cùng
budget; elapsed result gồm trace kết thúc nhưng không gồm ghi dòng result ra
đĩa. Chưa có hard timeout hay deadline end-to-end gồm ASR/I/O chính thức.

## Cấu trúc và bàn giao

| Vị trí | Chức năng |
|---|---|
| `src/lifemate/runtime/contracts.py` | Kiểu dữ liệu và Protocol tạm thời |
| `src/lifemate/runtime/runner.py` | Vòng điều phối, budgets, ask-stop |
| `src/lifemate/runtime/deadlines.py` | Deadline monotonic được inject clock |
| `src/lifemate/runtime/tracing.py` | JSONL, serialization và redaction |
| `src/lifemate/dev/` | Mock model/environment, fake clock và demo |
| `scripts/ops/demo.py` | CLI từ source, không cần cài package |
| `tests/ops/test_runtime.py` | Unit/integration tests offline |
| `docs/ops/CONTRACTS.md` | Các điểm cần LLM chốt và đường tích hợp |
| `docs/ops/CHECKLIST.md` | Phần đã kiểm tra và phần còn thiếu |
| `packaging/README.md` | Checklist packaging, chưa có Docker submission |

Kế hoạch gốc không bị sửa. Workspace scaffold này không có bản kế hoạch local;
đã đọc bản `LifeMate/PROJECT_PLAN_2026-11-03.md` từ repo chung trước triển khai.
Không có AGENTS.md áp dụng được tìm thấy trong workspace/các thư mục cha đã kiểm.

Đã kiểm tra bằng mock: truyền observation, thứ tự calls, isolation giữa samples,
không replay history, ask-stop, schema/parse repair, fake deadline, bảo toàn calls
khi lỗi, unknown execution không retry, budgets và JSONL/manifest.

Đã kiểm prompt/parser LLM và source runtime đã nhận bằng fixture. Chưa kiểm chứng
live LLM server, runtime/split chính thức, ASR, dữ liệu chính thức, evaluator,
latency A30, dependency GPU và submission.
Không có model/data download hoặc training. Adapter LLM dùng transport được inject từ bên LLM.
Adapter bước đầu đã có trong `src/lifemate/adapters/`; cần tài nguyên đúng split để chạy thật.
Thay `ScriptedModel` bằng adapter triển khai `ModelBackend.respond`; thay factory
`MockEnvironment` bằng adapter `Environment` sau khi đọc source ToolEnv chính
thức. Prompt builder/parser do bên LLM cung cấp. Không import `lifemate.dev`
vào đường submission tương lai.

## CI tự động

Workflow `.github/workflows/repository-ci.yml` ở gốc repo chạy khi push,
pull request hoặc thủ công. Job Ops dùng Ubuntu/Python 3.11, kiểm config,
chạy unittest và demo mock. Log và kết quả mock lưu 7 ngày. Workflow còn có
các job DATA, Speech và LLM riêng; xem [hướng dẫn cấu trúc repo và CI](REPOSITORY_GUIDE.md).
CI không xác nhận model/ToolEnv thật hay chất lượng submission; chưa có CD.

## Kiểm tra trước bàn giao / submission

Đã có [hướng dẫn preflight](docs/ops/PREFLIGHT.md) và
[checklist bàn giao AI/DATA → OPS](docs/ops/CHECKLIST.md).

```sh
.venv/bin/python scripts/ops/check_config.py configs/ops/runtime.example.json --provenance configs/ops/provenance.example.json
.venv/bin/python scripts/ops/check_submission.py /path/to/submission.jsonl --expected-ids /path/to/expected_ids.jsonl
```

Manifest demo dùng template provenance đầy đủ; giá trị chưa biết là null và được
liệt kê ở missing_provenance. Submission checker chỉ kiểm cấu trúc/coverage,
không xác nhận schema arguments theo catalog hoặc tính xác thực calls/hash.
