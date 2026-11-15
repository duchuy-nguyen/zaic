# Tài nguyên chia sẻ riêng, không đưa vào folder GitHub

`assets.manifest.json` là metadata tên file/dung lượng/SHA256, KHÔNG phải weights. Nó khớp nguyên bản assets_large của baseline50,9. Không dùng bộ Medium của50,2 thay vào.

| Tài nguyên cần nhận riêng | Mục đích |
|---|---|
| models/qwen/Qwen3.5-4B-LifeMate-V2-Q4_K_M.gguf | Qwen3.5 đã merge adapter V2 và quantize Q4_K_M |
| models/asr/ cùng processor/tokenizer/config | PhoWhisper-large đã merge adapter, FP16 |
| private_data/retrieval_index.json | Bank TRAIN-fit cho TF-IDF; chứa ví dụ nên giữ private |
| private_data/split.json | Split/group IDs cho VAL, warmup và loại nguồn retrieval |
| private_data/train_references.json | Tham chiếu TRAIN để chấm VAL sau inference |
| runtime/llama-server.exe cùng DLL/license b11800 đúng manifest | Runtime Windows CUDA đã chốt |
| assets.manifest.json ở gốc private_assets | Manifest nguyên bản; kiểm đồng thời với bản metadata public |
| Dataset BTC train+public_test v3.2.0 | Mỗi người tự lấy; không nằm trong package Git |

Để tránh lẫn assets vào Git, khuyến nghị để chúng ngoài checkout rồi sửa config.local.json. Nếu để trong private_assets/ thì thư mục đã bị ignore, nhưng audit public-tree sẽ cố ý chặn việc xuất bản nguyên folder đang chứa assets.

Nếu đã nhận ZIP private dùng chuẩn ARCHIVE_MANIFEST.json của nhóm:

    python -B tools/unpack_assets.py --archive PATH_TO_PRIVATE_ZIP --checksums PATH_TO_PRIVATE_SHA256SUMS --dest PATH_TO_PRIVATE_ASSETS

Không mặc định ZIP Medium cũ có thể biến thành Large. Công cụ chỉ giải nén/kiểm hash; không tạo hoặc tải weights còn thiếu. Danh sách file chính xác xem assets.manifest.json. Mỗi ZIP giữ manifest riêng; không ghi đè file khác hash.

Optimizer/checkpoints/token cache/source data/traces/scores/submission JSONL và link Drive có quyền truy cập là tài nguyên riêng. Package này không chứa chúng và cũng không tạo ZIP weights mới. Không cần checkpoint optimizer để inference bằng merged weights.
