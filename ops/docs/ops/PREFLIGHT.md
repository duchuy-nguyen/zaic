# Manifest, config và submission preflight

Chạy từ workspace root bằng `.venv/bin/python`; không cần dependency ngoài stdlib.

## Kiểm config và provenance

```sh
.venv/bin/python scripts/ops/check_config.py configs/ops/runtime.example.json --provenance configs/ops/provenance.example.json
```

Runtime config bắt buộc đủ 5 field: sample_deadline_seconds, max_executed_calls,
max_steps, max_repairs, output_tokens_per_step. Không chấp nhận field lạ/trùng,
NaN/Infinity hoặc sai type; max_repairs cho phép 0, budgets khác phải dương.
Không chứa endpoint/authentication trong config này.

`provenance.example.json` là mẫu bàn giao, toàn bộ thông tin chưa biết để null.
Copy sang file riêng trong runs khi nhận artifacts; không điền secrets, URLs
có credential hay tên người dùng. Dùng revision/checksum/source identifier.
Validator chỉ kiểm cấu trúc/type/hash format; không xác minh các giá trị có đúng
với artifact thực tế hay không. Trạng thái valid không có nghĩa đã đủ provenance.

```sh
.venv/bin/python scripts/ops/demo.py --provenance configs/ops/provenance.example.json
```

Manifest mới gồm phiên bản schema, runtime config, UTC timestamp, OS/Python,
Git revision/dirty nếu có, fingerprint source/scripts/pyproject, data/split/catalog,
ToolEnv runtime, base/adapter/tokenizer/template/prompt, ASR, dependency/container,
hardware, seed/decode/precision/context/transcript, scorer/timing mode.
`missing_provenance` liệt kê field null; `provenance_complete` chỉ là đủ metadata,
không phải chứng nhận bài nộp hợp lệ. Demo luôn official_submission=false.
Null là đúng cho thông tin chưa bàn giao/không áp dụng; chưa có readiness gate
cho từng loại thí nghiệm. Manifest ghi rõ deadline hiện chỉ bao quanh runner.

## Kiểm submission chỉ đọc

Chuẩn bị `expected_ids.jsonl` từ DATA: mỗi dòng `{"id":"..."}`. Có thể dùng JSONL
dataset làm nguồn ID vì loader chỉ sử dụng field id, không log nội dung khác.
Nguồn ID phải độc lập, đúng split; không lấy từ chính submission để kiểm coverage.

```sh
.venv/bin/python scripts/ops/check_submission.py /path/to/submission.jsonl --expected-ids /path/to/expected_ids.jsonl
```

Exit code: 0 nếu structural check pass; 1 nếu submission lỗi; 2 nếu đầu vào/config
không đọc được hoặc nguồn ID không hợp lệ. JSON report ra stdout; không sửa file,
không execute ToolEnv, không gọi mạng hoặc upload. Report dùng số dòng/mã lỗi,
không in IDs, arguments hoặc nội dung dataset.

Kiểm UTF-8, JSONL (mỗi dòng một object; không blank line/BOM/duplicate JSON keys,
NaN/Infinity), đủ/không trùng/không lạ ID, đúng 3 field id/tool_calls/state_hash.
ID và hash phải là string không rỗng; tool_calls là list, mỗi call có name string
không rỗng và arguments object. Cho phép calls rỗng và nhiều calls trong kết quả
cả sample; giới hạn 1 call/lượt model không phải giới hạn 1 call/submission row.
Không giả định hash phải dài bao nhiêu/hex khi chưa có API official.

Mặc định giới hạn 5 MiB theo tài liệu nộp bài trong repo; `--max-bytes` cho phép
đổi khi hướng dẫn được cập nhật. Không enforce deadline 30s từ JSONL kết quả.

**Chưa kiểm:** schema arguments theo catalog, tool permissions, calls/hash có
thực sự lấy từ đúng ToolEnv session, semantic/task quality. Các phần này cần
ToolEnv/catalog/evaluator thật. Structural pass không chứng minh official validity.
Không sinh submission mẫu trong source hoặc gọi output mock là bài nộp.
