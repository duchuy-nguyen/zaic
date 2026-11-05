# LifeMate — khung Ops

Chép các thư mục bên trong gói vào thư mục LifeMate của repo. Không thay bản kế hoạch hiện có.

Đây là scaffold, chưa có runner, model, ASR, evaluator hoặc ToolEnv implementation. Không phải solution chạy được hay Docker submission hoàn chỉnh.

## Phạm vi
- `src/lifemate/runtime/`: điều phối, deadline, trace và adapter ToolEnv.
- `configs/ops/`: cấu hình runtime sau khi thống nhất với nhóm.
- `scripts/ops/`: CLI kiểm tra môi trường và chạy pipeline sau tích hợp.
- `packaging/`: Docker/entrypoint offline sau khi xác nhận hướng dẫn BTC.
- `tests/ops/`: kiểm tra hành vi runtime khi triển khai.
- `docs/ops/`: hướng dẫn setup, interface và checklist bàn giao.

## Thứ tự triển khai
1. Tạo môi trường Python 3.11 biệt lập.
2. Thống nhất interface model và transcript provider cùng nhóm.
3. Nhận ZIP chính thức, đọc README/API ToolEnv trước khi viết adapter.
4. Triển khai runner và trace, kiểm tra bằng mock trong tests.
5. Kiểm tra với ToolEnv thật rồi tích hợp model cho B0A.
6. Khóa dependency sau smoke; hoàn thiện Docker offline khi stack ổn định.

Không commit dữ liệu, transcript, state, model weights, secrets hoặc runs.
