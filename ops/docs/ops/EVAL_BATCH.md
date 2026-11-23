# VAL evaluation và resumable public batch

Khung chạy trước khi có GPU; chưa có chứng nhận live inference/runtime Linux.
Giữ baseline50 FIT1236/VAL309, không đổi sang split DATA1237/308. Input là data
BTC, không phải synthetic của nhóm. VAL là split đã được development ở baseline
gốc, không trình bày điểm VAL như một held-out test chưa từng dùng để phát triển.

## Preflight không model

```sh
python ops/scripts/ops/run_baseline50_batch.py val --config /private/config.local.json --output-dir /private/val_run --preflight-only
python ops/scripts/ops/run_baseline50_batch.py public --config /private/config.local.json --output-dir /private/public_run --preflight-only
```

Config là bốn path của baseline: assets_dir, dataset_dir, output_dir, llama_server.
Preflight chỉ cần private split/index, dataset/catalog/SDK/audio references.
Chạy canonical loader để kiểm SHA JSONL/catalog/toàn bộ Python SDK, public
input fields, đủ public634, đúng VAL309. Input model được allowlist, bỏ label và
audio_text, giữ assistant history có sẵn, không chạy lại history. Kiểm audio
path tồn tại và nằm trong split; không đọc state hoặc tạo ToolEnv. Không đo
audio quality hay chứng nhận assets model từ preflight.

## Chạy thật khi môi trường đã sẵn sàng

```sh
python ops/scripts/ops/run_baseline50_batch.py val --config /private/config.local.json --output-dir /private/val_run
python ops/scripts/ops/run_baseline50_batch.py public --config /private/config.local.json --output-dir /private/public_run
```

Hai split phải chạy process riêng. Factory pin toàn bộ weights/runtime, kiểm
đúng SDK import, giữ frozen policy/ASR batch và tạo đúng một ToolEnv cho mỗi mẫu
được chạy mới. Không dùng factory FIT smoke để chạy VAL/public. Không tự warmup
trên VAL/public; startup không nằm trong latency mỗi sample. Các tham số policy
giữ theo bridge baseline50, chưa thêm semantic guards hoặc train/tune public.
Runtime hiện pin Windows CUDA b11800; Mac không chạy được, Linux/A30 cần variant
riêng được kiểm chứng. Source model gốc không thay đổi.

Outputs private: run_manifest.json, records/000000.json..., inflight.json nếu
gián đoạn, submission.jsonl, diagnostics.jsonl và submission_check.json.
Không tự upload. Full partition IDs đến từ canonical loader độc lập với output,
không lấy expected IDs từ chính submission. References chỉ được evaluator mở
sau inference, không nhập vào factory/model. Mọi call kể cả call bị từ chối và
hash giữ nguyên output SDK, không redaction/reconstruct danh sách call để chấm.

## Resume an toàn

Thêm `--resume` vào chính lệnh cũ. Không đổi source/config/paths/input/assets.

- Lock OS không cho hai process sở hữu cùng run. Sau crash, OS tự thả lock.
- Atomic marker được lưu **trước** khi mở session. Nếu mẫu chưa có committed
  record, resume chặn và yêu cầu điều tra, không tự chạy lại hoặc reset ToolEnv.
- Record có ordinal/ID/hash, phải là prefix liên tục, được kiểm trước load model.
- Có record nhưng marker còn lại sau crash: xác nhận ID/ordinal và bỏ marker,
  tiếp tục đúng mẫu kế tiếp; không thực thi lại mẫu đã commit.
- Đủ records: chỉ hoàn thiện output, không khởi tạo model hoặc session mới.
- Không ghi đè output khác nội dung, không retry mẫu đã ghi nhận failure.

Không xóa marker thủ công để làm resume vượt kiểm tra. Nếu process chết giữa
thực thi và commit, không thể chứng minh an toàn để tự chạy lại phiên đó; giữ
artifact và điều tra thay vì tuyên bố exactly-once qua mọi sự cố phần cứng.
Journal fsync file và atomic replace; không phải bảo đảm phục hồi mọi lỗi disk.

## Chấm VAL offline

```sh
python ops/scripts/ops/evaluate_baseline50_val.py --dataset-dir /private/train+public_test --private-data /private/private_assets/private_data --predictions /private/val_run/submission.jsonl --diagnostics /private/val_run/diagnostics.jsonl --output-dir /private/val_evaluation
```

Có thể truyền historical traces.jsonl làm predictions và bỏ diagnostics vì trace
đã có status/elapsed_s. Dùng scorer frozen local-spec-v1, references đúng hash
pin, tokenizer word theo baseline. Đây là local approximation, không phải điểm
BTC; không thay bằng DATA scorer default whitespace rồi so trực tiếp.

Reject duplicate/unknown IDs. Missing mẫu được giữ trong denominator và nhận 0;
summary.complete=false, CLI exit1. Timeout, lỗi điều phối và latency>30s nhận 0,
kể cả gold-call rỗng. Call-budget stop giữ formula baseline, không tự gán failure.
Phải có status và latency từng prediction, không suy thành công từ JSONL ba field.
Lưu scores.json riêng với summary.json và source/reference/input hashes.

Metrics: score100, toolF1, state match, missing/failed/statuses, over30s và
p50/p95/max latency theo số mẫu có execution record. Latency là field đã ghi,
không chứng minh đủ audio I/O → inference → disk output. ops_run_wall_s trong
record bao quanh system.run, chưa gồm ghi journal/serialization ra disk; timing
BTC end-to-end và VRAM vẫn phải đo thực tế trên GPU. Không gắn nhãn timing
fixture hay historical timing là kết quả chạy mới.

## Đã kiểm local

- Preflight official VAL309 và public634 thành công, không load model/runtime.
- Chấm lại 309 historical predictions bàn giao: 57.6539084791512/100, khớp
  từng score baseline lịch sử; không phải lượt inference mới.
- Fixture tests kiểm resume prefix, uncommitted block, marker còn sau commit,
  input drift, corrupt checkpoint, coverage, cleanup, missing/timeout scoring.
- Chưa chạy hai factory VAL/public bằng weights/GPU thật; chưa Docker rehearsal.
