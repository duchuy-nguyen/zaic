# Runtime thuộc OPS

Đã có contracts.py, runner.py, deadlines.py và tracing.py cho demo offline.
Contracts là đề xuất nội bộ, chưa phải API LLM server hoặc ToolEnv.
Mock chỉ nằm ở `lifemate.dev`; runtime không import mock.

Chưa viết ToolEnv bridge, prompt builder/parser, HTTP transport, ASR, evaluator
hay official submission writer. Xem README gốc và docs/ops/CONTRACTS.md để tích hợp.
