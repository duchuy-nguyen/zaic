# Checklist packaging — chờ integration

Chưa tạo Dockerfile/entrypoint submission. Demo mock không phải submission.

- [ ] Nhận hướng dẫn I/O, entrypoint, paths và offline policy chính thức.
- [ ] Chốt model/ASR/server CLI và weights/tokenizer revisions cùng nhóm.
- [ ] Smoke stack thật, khóa dependencies, wheels và container digest.
- [ ] Pin ToolEnv đúng split; không thay source official hoặc đọc state/labels ở agent.
- [ ] Đảm bảo submission không import mock và không dùng mock_results làm official output.
- [ ] Lấy executed calls/hash từ session thật theo API đã xác nhận.
- [ ] Kiểm offline/no-network trên target hardware và deadline end-to-end.
- [ ] Theo dõi partial failures, output coverage, reproducibility và model-load timing.
