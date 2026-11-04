# Interface cần thống nhất

Các nội dung sau là đề xuất nội bộ, chưa phải API của ToolEnv.

| Thành phần | Input | Output | Phụ trách |
|---|---|---|---|
| Transcript provider | Audio refs current/history | Text + provenance + latency | ASR |
| Model backend | Messages, schemas, deadline, token budget | Raw output, proposals hoặc finish, usage | LLM + OPS |
| Runtime bridge | Proposed call | Validation, executed flag, observation | OPS |
| Run artifacts | Config, sample và step events | Trace, calls/hash, timing, failure | OPS |

Runtime tạo một ToolEnv mỗi mẫu, thực thi tuần tự, không replay history; stop sau executed ask. Inference không đọc label/final_state hoặc gọi get_state(). Submission lấy calls/hash từ session thật.

Thống nhất kiểu lỗi, cách báo finish và định dạng observation trước khi triển khai. Chưa giả định constructor/signature ToolEnv khi chưa có source chính thức.
