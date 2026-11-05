# ZAIC — LifeMate

Các phần triển khai nằm ngang hàng ở gốc repo:

| Thư mục | Nội dung |
| --- | --- |
| `data/` | Pipeline DATA, manifests và assets synthetic đã commit |
| `speech/` | PhoWhisper, audit audio, transcript và ASR evaluation |
| `llm/` | Agent, backend, scoring và thí nghiệm LLM |
| `ops/` | Điều phối, budgets/deadline, logging và preflight |
| `LifeMate/` | Kế hoạch, deadlines và vị trí tài nguyên local LLM hiện có |
| `scripts/ci/` | Kiểm tra source và assets DATA, không khởi chạy pipeline |
| `.github/workflows/` | GitHub Actions, bắt buộc ở gốc repo |

## CI

`Repository CI` chạy trên push, pull request và thủ công, Python 3.11/Ubuntu.
Bốn job độc lập: DATA (cú pháp và cấu trúc assets), Speech (tests với fixture,
CPU torch), LLM (tests độc lập dữ liệu riêng) và Ops (tests/config/demo).
Logs được lưu 7 ngày. Không tải model, official dataset, không gọi inference
server, train, submit hoặc deploy. Thư viện CPU phục vụ tests được cài từ mạng;
model access của Hugging Face được đặt offline trong job Speech.

LLM CI chạy `test_baseline`, `test_analysis`, `test_diagnostic`,
`test_extra_metrics`, `test_gemma`. `test_expanded` và `test_full` cần manifest,
dataset và snapshot thí nghiệm riêng nên chưa nằm trong CI công khai. DATA
chưa có bộ unittest riêng; structural checks không thay cho replay/evaluator.
CI xanh không xác nhận end-to-end với model/ToolEnv chính thức hoặc latency A30.

## Đường dẫn sau khi sắp xếp

- `LifeMate/speech/` → `speech/`.
- `LifeMate/{src,scripts,manifests,synthetic}/` → `data/` với cấu trúc con giữ nguyên.
- `ops/` và `llm/` giữ vị trí hiện tại.
- Không chuyển model, dataset hoặc artifacts riêng. LLM vẫn dùng `LifeMate/`
  làm resource root mặc định; có thể đặt `LIFEMATE_ROOT` theo hướng dẫn LLM.
- Speech: dùng `LIFEMATE_SPEECH_WORK_DIR` để tiếp tục dùng cache/output cũ.
  Nếu không đặt biến này, mặc định mới là `speech_work/` ở gốc repo.

Chi tiết: [DATA](data/README.md), [Speech](speech/README.md),
[LLM](llm/README.md), [Ops](ops/README.md),
[kế hoạch](LifeMate/PROJECT_PLAN_2026-11-03.md).
