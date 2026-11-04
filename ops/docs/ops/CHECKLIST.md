# Checklist Ops nền tảng

## Đã kiểm tra bằng mock

- [x] Demo offline CPU, stdlib, môi trường project Python 3.11 riêng.
- [x] Contract nội bộ phân biệt calls/finish/error; mock chỉ ở `lifemate.dev`.
- [x] Factory tạo environment riêng, thực thi tuần tự, không replay history.
- [x] Truyền observation vào bước sau, lookup ID rồi update.
- [x] Stop sau ask đã execute, calls rỗng hợp lệ, lỗi hệ thống, budgets, deadline.
- [x] Tối đa một call/lượt; parse/schema repair qua LLM và budget chung toàn sample.
- [x] ok=false giữ observation về model, không tự retry hoặc mất executed calls.
- [x] Deadline toàn vòng runner bằng monotonic/fake clock; không sleep trong tests.
- [x] Giữ executed calls khi lỗi sau đó, unknown execution không retry.
- [x] `done=false` không bị chuyển thành task success.
- [x] Mỗi demo sample có result; trace JSONL, manifest và source fingerprint.
- [x] Tên output mock; không có hash state giả hoặc submission.jsonl.
- [x] Secret keys được redaction; exception text/input/history không vào logs.

## Chờ integration / chưa kiểm chứng

- [ ] LLM chốt request/response, parser, observation và endpoint.
- [ ] Nhận source chính thức, đọc API ToolEnv và pin split/runtime revision.
- [ ] Adapter thật: session/calls/hash, unknown execution semantics, no leakage.
- [ ] Hard timeout/cancellation và timing end-to-end gồm ASR, I/O, output.
- [ ] Model/ASR/data/catalog/template provenance, dependency lock sau smoke.
- [ ] B0A/B0–B3 contracts/budgets freeze cùng nhóm.
- [ ] Evaluator do DATA phụ trách; không suy chất lượng từ mock pass.
- [ ] Offline packaging và A30 rehearsal theo checklist packaging.

## Checklist bàn giao AI / DATA → OPS

### Đã chuẩn bị phía Ops

- [x] Runtime config loader kiểm đủ field, không nhận field lạ/trùng, sai type hoặc budget không hợp lệ.
- [x] Manifest version 1 có provenance template và danh sách thông tin còn thiếu, không tạo giá trị giả.
- [x] CLI structural preflight submission: UTF-8 JSONL, ID đủ/không trùng/không lạ, schema dòng/call và giới hạn dung lượng.
- [x] Checker chỉ đọc; không sinh hash, execute tool, sửa submission hay upload.

### AI bàn giao

- [ ] Model/base/adapter/tokenizer revision; logical parameter count và giấy phép/provenance.
- [ ] Chat template/prompt cùng SHA256; parser và cách render RepairFeedback.
- [ ] Request/response version, server command/port/endpoint và offline assets.
- [ ] Decode/seed/precision, context limit và kiểm input+output token budget.
- [ ] Smoke response 0/1 call, parse error, validation repair và observation ok=false/done=false.

### DATA bàn giao

- [ ] Archive SHA256, split name/manifest SHA256, ID list và catalog SHA256.
- [ ] Source ToolEnv nguyên bản đúng split, runtime revision/hash và API đã xác nhận.
- [ ] Sample input sạch labels/state; history và transcript source rõ ràng.
- [ ] Evaluator/scorer version và references train-only tách inference.

### Ops xác nhận sau khi nhận

- [ ] Điền provenance thật; kiểm file hashes/revisions với artifacts nhận được.
- [ ] Chạy check_config với runtime config + provenance, giải quyết missing_provenance.
- [ ] Adapter thật, calls/hash cùng session; không dùng mock_results làm submission.
- [ ] Argument schema/tool permissions kiểm bằng catalog/ToolEnv thật; structural checker không thay phần này.
- [ ] Đo end-to-end deadline, offline smoke và target hardware; lock dependencies/container digest.
- [ ] Chạy checker với danh sách ID đúng split; lưu kết quả rồi review task quality riêng.
