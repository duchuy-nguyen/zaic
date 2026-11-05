# Cấu hình Ops development

`runtime.example.json` được CLI demo đọc thành `RuntimeConfig`.
Chỉ có deadline, max executed calls, max model steps, max_repairs (chung toàn sample, mặc định 2, cho phép 0) và output token budget.
Tất cả là giới hạn thử nghiệm, chưa freeze B0–B3 hoặc kiểm chứng bằng model.
Không có URL/port/server command/weights/tokenizer vì interface LLM chưa chốt.
Không đặt secrets hoặc đường dẫn dữ liệu riêng vào config.
