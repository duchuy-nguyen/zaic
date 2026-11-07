# Runtime thuộc OPS

Đã có contracts.py, runner.py, deadlines.py và tracing.py cho demo offline.
Contracts là đề xuất nội bộ, chưa phải API LLM server hoặc ToolEnv.
Mock chỉ nằm ở `lifemate.dev`; runtime không import mock.

Đã có ToolEnv bridge và native LLM adapter trong `lifemate.adapters`; chúng tái
sử dụng prompt/parser của LLM lane. Khung `lifemate.system` bổ sung prepared
text/audio boundary, retrieval, grounding và export calls/hash từ cùng session.
Xem `docs/ops/SYSTEM_SCAFFOLD.md`. ASR/model/official split cần adapter và kiểm
chứng tích hợp riêng; demo fixture không phải chứng nhận submission chính thức.
