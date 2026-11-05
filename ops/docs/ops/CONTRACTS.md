# Contract nội bộ tạm thời — chờ xác nhận

Cập nhật 05/11: đã thêm bridge cho API runtime DATA đã commit và prompt/parser
LLM trong `src/lifemate/adapters/`. Có 11 tests tích hợp source với fixture;
chưa xác nhận runtime train/public chính thức hoặc live model. Chi tiết ở
[HANDOFF_REVIEW_2026-11-05.md](HANDOFF_REVIEW_2026-11-05.md). Các mục chờ xác nhận
bên dưới vẫn áp dụng cho live deployment, timing và split chính thức.

Đây là kiểu Python cho demo Ops, không phải API HTTP, prompt format hay signature
ToolEnv chính thức. Không freeze B0–B3 từ mock này.

| Kiểu | Ý nghĩa |
|---|---|
| `Sample` | ID và prepared model input; không truyền label/final_state |
| `ModelRequest` | Sample, step, observations tích lũy, remaining seconds, output budget, repair và repairs_remaining |
| `RepairFeedback` | Loại parse/schema/protocol error, proposal và validation nếu có; đưa về LLM sửa |
| `ProposedCall` | Tên và arguments đề xuất; chưa xác nhận đã execute |
| `Calls` / `ModelError` | Calls rỗng hợp lệ là finish; một call là action; ModelError phân biệt parse/model error |
| `Validation` | Hợp lệ, classification ask, code; validate không đổi môi trường |
| `Execution` | executed=true/false/unknown và observation |
| `Observation` | Giữ ok, done, data; terminal chỉ là metadata, tool error vẫn về model |
| `SampleResult` | Executed calls, uncertain calls, last observation, stop reason, orchestration failure |

Backend nhận snapshot riêng của sample/observations để không vô tình sửa lịch
sử thực thi trong runner. Environment adapter xác định tool nào là ask từ API
đã xác nhận; runner không đoán tên tool official. Trong mock chỉ có `demo.ask`.
Factory sở hữu việc tạo session mới mỗi sample; không share singleton, không
replay assistant history, không reset/retry session để chọn kết quả tốt hơn.

## Chính sách nhóm đã thống nhất

- Mỗi response parse thành công có 0 hoặc 1 call: 0 kết thúc, 1 validate rồi
  execute. Nhiều calls là protocol error; không execute bất kỳ call nào.
- Adapter chỉ tạo `Calls(())` từ output rỗng **hợp lệ, parse thành công**. Không
  map parse exception/timeout thành calls rỗng; không dùng kiểu Finish riêng.
- Parse/schema/protocol rejection gửi `ModelRequest.repair` về backend LLM ở
  lượt tiếp. Ops không dựng prompt/parser, đoán arguments hay retry action.
- `max_repairs` là số lượt repair tối đa toàn sample (mặc định 2, cho phép 0),
  dùng chung parse/schema/protocol, không reset sau call thành công. Hết budget
  ghi `repair_budget` và failure. `repairs_remaining` được gửi cho backend.
- `repairs_used` đếm lượt đã lên lịch; `max_steps` và deadline có thể ngăn lượt
  đó bắt đầu. Mỗi lần backend, kể cả repair, tính một step.
- Reject trước execute không vào executed calls. `ok=false`, `done=false` vẫn
  được đưa nguyên về model, không tự retry. Trạng thái execution unknown dừng
  failure ngay, giữ previous calls; không đưa vào vòng repair có thể replay.
- `sys.ask_missing_info` dừng chỉ sau execute; mock dùng `demo.ask` qua is_ask.
- Finish/ask là dừng điều phối, không xác nhận task success. Deadline và call
  budget áp dụng xuyên suốt kể cả repair, không reset environment hoặc timer.

## Các điểm cần LLM chốt

| Điểm | Đề xuất demo / câu hỏi cần xác nhận |
|---|---|
| Request/response | Ai chuyển messages/tools thành transport? Chính xác field, version, raw output/usage và finish representation là gì? |
| Calls | Đã chốt tối đa 1 call/lượt; cần LLM xác nhận mapping wire format sang Calls(()) hoặc Calls((call,)) |
| Prompt/parser/error ownership | LLM cung cấp prompt builder/parser. Parse error khác finish; Ops đã có repair feedback/budget; LLM cần cung cấp error rendering/diagnostics và thực hiện repair, không retry tool tự động. |
| Observation format | Giữ ok/done/data/terminal trong demo; cần mapping từ ToolEnv thật, format messages/tool IDs và báo execution unknown |
| HTTP endpoint | Chưa có port, URL, authentication, lệnh server hay weights; chưa viết HTTP client |
| Deadline/cancellation | Backend nhận remaining seconds; cần chốt timeout transport/cancel và chi phí prompt/tokenization, không xem cooperative deadline là hard timeout |
| Token/call budgets | Hiện chỉ cấu hình development; chốt context+output budget và policy trước freeze |

## Thay mock sau này

1. Bên LLM viết adapter `ModelBackend.respond(request)`, dùng prompt builder và
   parser do LLM sở hữu. Không cho framework tự execute tool. Không tạo wire
   JSON từ dataclass rồi mặc định đó là protocol server.
2. Ops đọc source/README ToolEnv chính thức trước khi viết bridge `Environment`.
   Giữ nguyên schema/extensions; không đoán constructor hoặc method signature.
   Validate side-effect free, execute tối đa một lần; xác định executed/unknown
   theo bằng chứng thật, không đoán từ `ok`.
3. Inject factory tạo đúng một ToolEnv/sample; pin runtime/split, tách process
   train/public khi cần. Agent không đọc state snapshot/labels hoặc get_state().
4. Submission writer riêng phải lấy calls/hash từ session thật theo API được
   xác nhận. Danh sách call trong demo không thay nguồn official; không tự tạo
   state hash. Mọi output hiện tại chỉ là `mock_results.jsonl`.
5. Kiểm integration với data/model thật và target timing trước B0A; lock
   dependencies sau compatibility smoke. Chưa triển khai các bước này.
