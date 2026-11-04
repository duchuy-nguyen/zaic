# Kết quả kiểm tra Ops mock

Ngày kiểm tra: 04/11/2026 (GMT+7).

## Môi trường

macOS arm64, CPython 3.11.18, `.venv` riêng trong workspace. Python mặc định máy
là 3.9.6 và không bị sửa. Vì máy chưa có Python 3.11, đã dùng bản portable
python-build-standalone release 20261026 đặt trong `/private/tmp/lifemate-python311`
để tạo venv. SHA256 archive đã đối chiếu với digest GitHub release:
`f3e1cd83a7f1d70e58b0d69102c18bd6867d2f973068dc5cbfd5822e445c66df`.
Không cài thêm runtime/test dependencies hoặc global packages. Venv này tham
chiếu Python portable trong thư mục tạm; nếu thư mục tạm bị dọn, tạo lại venv
bằng Python 3.11 có sẵn theo README. Không commit/copy venv để bàn giao.

## Lệnh và kết quả

```sh
.venv/bin/python --version
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/ops -v
.venv/bin/python scripts/ops/demo.py
```

Kết quả: Python 3.11.18; **50 tests pass**. Config JSON và pyproject TOML parse
thành công; project dependencies rỗng. Đã chạy demo độc lập ngoài test suite,
đủ 8 mẫu, stop reasons/call counts đúng bảng README.

Run kiểm tra cuối:
`runs/mock_20261104T125405_176954Z/`.
Có `run_manifest.json`, `tool_traces.jsonl`, `mock_results.jsonl`;
không có submission.jsonl/state hash giả. Các artifacts chạy nằm trong runs
bị gitignore, không phải source cần commit. Integration test cũng kiểm đủ
8 result/stop events, config trong manifest, redaction, và chặn ghi đè run cũ.

Tests deadline dùng fake clock, không sleep. Đã kiểm các điểm backend trả trễ,
factory hết budget, validation hết budget, logging tiêu tốn budget, execution
hoàn tất trễ, và giảm thời gian còn lại giữa các lượt. Không có bằng chứng về
hard timeout: hàm đang block không bị ngắt.

## Phạm vi thay đổi

- Thêm runtime: `contracts.py`, `deadlines.py`, `runner.py`, `tracing.py`.
- Thêm development mocks/demo: `src/lifemate/dev/{__init__,mocks,demo}.py`.
- Thêm CLI `scripts/ops/demo.py` và tests `tests/ops/test_runtime.py`.
- Cập nhật `pyproject.toml`, `.gitignore`, runtime example config.
- Cập nhật README gốc, README runtime/config/scripts/tests/packaging/lockfiles,
  `docs/ops/CONTRACTS.md`, `docs/ops/CHECKLIST.md`; thêm báo cáo này.

Kế hoạch đọc từ repo chung chỉ được tải để đọc vào thư mục tạm, không sửa.
Workspace không có Git metadata; không khởi tạo repo, commit, push, merge hoặc
sửa remote trong tác vụ triển khai này.

Đã áp dụng tối đa một call/lượt, repair budget chung parse/schema/protocol,
parsed calls rỗng kết thúc, tool error đưa về model và ask đã execute thì dừng.
Tests mới kiểm repair feedback, argument không bị Ops sửa, giữ calls sau lỗi,
không reset repair budget sau execution, timeout/deadline trong repair,
multi-call bị từ chối trước execute và sys.ask_missing_info.

Chờ bên LLM chốt wire interface, prompt/parser/observation và
HTTP endpoint; chờ source ToolEnv/dữ liệu chính thức để viết adapter và kiểm
integration. Chưa triển khai model/ASR/training/evaluator hay Docker submission.


## Bổ sung manifest/config/submission preflight

Đã thêm configuration.py (strict JSON/config), manifest.py (schema provenance),
submission_check.py (read-only structural validator), hai CLI check_config.py /
check_submission.py, provenance.example.json và tests/ops/test_preflight.py.
Hướng dẫn ở PREFLIGHT.md; checklist bàn giao được cập nhật.

Lệnh kiểm tra bổ sung:

```sh
.venv/bin/python scripts/ops/check_config.py configs/ops/runtime.example.json --provenance configs/ops/provenance.example.json
```

Config/provenance template pass; trường chưa biết để null, không xác nhận đã
bàn giao đủ. Test suite tổng 50 tests, gồm CLI exit codes 0/1/2, ID coverage,
duplicate/unknown IDs, malformed JSON/UTF-8, size boundary, strict config,
provenance completeness và không ghi nội dung dataset ra report.
Chưa kiểm bằng submission/catalog/ToolEnv thật; không push hoặc sửa remote.
