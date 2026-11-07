# DATA — LifeMate

Bản đã lọc 311 mẫu cho pipeline text-training: xem
[release, exclusions và cách kiểm checksum](releases/synthetic_phase1_v2_filtered_311/README.md).
Bản này loại 20 mẫu từ merge v2; không thay dữ liệu gốc hoặc chứng nhận nhãn
đúng toàn bộ.

Chạy lệnh DATA từ thư mục `data/`, đặt `PYTHONPATH=src` khi dùng
`python -m lifemate.synthetic.<module>`. Code được chuyển nguyên trạng từ
`LifeMate/src`, `LifeMate/scripts`, `LifeMate/manifests`, `LifeMate/synthetic`.

- `src/lifemate/synthetic/`: pipeline tạo và kiểm tra dữ liệu tổng hợp.
- `scripts/extract_official.py`: giải nén dữ liệu local; không chạy trong CI.
- `manifests/`: manifest train-fit tạm thời đã commit.
- `synthetic/`: assets tổng hợp và bản runtime đi kèm đã có trong repo.
- `official/`, `runs/`, `synthetic/synthetic_environment/`: tài nguyên local,
  bị gitignore. Nếu đã có tài nguyên ở `LifeMate/`, cần cấu hình/bố trí lại
  đường dẫn local trước khi chạy pipeline; việc chuyển source không chuyển dữ liệu riêng.

CI chỉ kiểm cú pháp Python, JSON/catalog, ID duy nhất và số lượng theo metadata.
Chưa chạy replay/kiểm leakage với dữ liệu chính thức. Runtime trong `synthetic/`
không được xem là runtime train/public-test đã xác minh cho submission.
