# Đúng hai thành phần đã tạo baseline50,9

- qwen35_training.py: nhánh LLM của train_v2.py gốc, chọn cố định Qwen3.5-4B; không còn nhánh ASR-medium.
- phowhisper_large_training.py: nhánh ASR của train_worker.py gốc, chọn cố định PhoWhisper-large; không còn nhánh Qwen3.
- completion_loss.py và tokenization.py: chỉ trích đúng các hàm collation, completion-only CE và native template được sử dụng; không mang theo generic Kaggle/Unsloth trainer.
- qwen_training_policy.py / asr_training_policy.py và efficient_attention.py: helper liên quan của source gốc.
- Recipe chuẩn hóa hai model được deploy ở ../training_recipe.json. Qwen489 update, ASR250 update. Training decode ASR batch4 khác deployment serial batch2.

Đây là bản trích source tham khảo đã chọn nhánh, KHÔNG phải CLI training portable đã được thử train lại. Hàm training giữ logic nhánh tương ứng; chỉ bỏ nhánh model không dùng và đổi import helper cho package. Không có main tự chạy. Import không nạp model hay bắt đầu training.

Muốn gọi training cần bind_private_workspace(path, workspace_root) đến workspace private đã khôi phục đủ manifest, data, token cache, config và model snapshots. Hàm bind kiểm đúng model. Không dùng các source đã tách để tiếp tục checkpoint gốc: source/manifest hash khác. Không bỏ guard hash để ép chạy.

Recipe/dữ liệu/optimizer/weights thật nằm riêng. Source đầy đủ chưa chọn nhánh vẫn được bảo toàn ở thí nghiệm gốc local; không đưa bản mixed đó vào package public này. DERIVATION.json ghi source hash và các hàm đã trích, không chứa đường dẫn tuyệt đối hoặc dữ liệu mẫu.

Inference dùng ../baseline50 và không import training_reference. Chưa thực hiện training mới hoặc kiểm numerical parity của toàn bộ vòng train đã tách; các kiểm tra ở đây là cú pháp, chọn đúng nhánh, import không GPU và đối chiếu source. Inference25tests vẫn chỉ xác nhận CPU/đóng gói, không phải lần chấm50,9 mới.
