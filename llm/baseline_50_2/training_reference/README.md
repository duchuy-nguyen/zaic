# Source lịch sử phục vụ audit

Các file trong thư mục này là bản sao source huấn luyện/chuẩn bị dữ liệu đã tạo ASR medium V3 và Qwen V2. Không chứa weights, audio, transcript hoặc ví dụ thực.

- `qwen_v2_train.py`, `qwen_v2_prepare.py`: QLoRA/next-action SFT và chuẩn bị dữ liệu V2.
- `medium_v3_train.py`, `medium_v3_prepare.py`: source experiment V3 gốc, có cả các nhánh Qwen/ASR lịch sử; model dùng trong baseline là nhánh ASR checkpoint250.
- Helper loss nếu có nằm cùng đây để đọc thiết kế completion-only loss.

Đây không phải entrypoint train độc lập: source phụ thuộc layout project cũ, kernel/toolchain và prepared data. Không chạy trực tiếp trong gói inference này để đoán rằng có thể resume an toàn. Recipe dùng thật nằm ở `../training_recipe.json`; checkpoint đầy đủ và indexes dữ liệu nằm trong ZIP Drive training. Khôi phục training cần base pretrained/pinned version và data/kernel phù hợp, rồi kiểm checkpoint/split trước khi cập nhật optimizer.

Entrypoint chạy baseline của bản phát hành là `python -m baseline50`; nó không gọi các file lịch sử này.
