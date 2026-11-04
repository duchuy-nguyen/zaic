# Checklist Ops

- [ ] Môi trường Python 3.11 biệt lập; ghi OS/GPU/CUDA thực tế.
- [ ] Model/ASR/template/runtime revision và data hashes có provenance.
- [ ] ToolEnv train/public tách process hoặc import isolation rõ ràng.
- [ ] Runner một session/mẫu; validate trước execute; stop sau ask.
- [ ] Deadline monotonic toàn mẫu; không thực thi call đến muộn.
- [ ] Trace phân biệt proposal và executed call; không mất calls sau lỗi.
- [ ] Mỗi ID có output hoặc failure; không âm thầm bỏ mẫu.
- [ ] Không label/state leakage; không commit dữ liệu riêng.
- [ ] Manifest và cấu hình cho phép chạy lại thí nghiệm.
- [ ] Lock dependencies sau compatibility smoke.
- [ ] Docker offline và entrypoint đối chiếu hướng dẫn BTC.
- [ ] Timing thực trên target hardware; không dùng cache để tuyên bố latency.
