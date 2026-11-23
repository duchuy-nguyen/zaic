# Chuẩn bị smoke từ bộ bàn giao baseline50

Lệnh chuẩn bị chạy offline bằng stdlib Python 3.11+, không import ASR/model,
không chạy exe và không thay đổi SDK hoặc state. Đây là luồng **BTC TRAIN**,
không phải synthetic của nhóm. Chỉ chuẩn bị các ID FIT của baseline50, không
nhận DATA split-v1 làm thay thế split baseline.

```sh
python ops/scripts/ops/prepare_baseline50_smoke.py \
  --dataset-dir /private/train+public_test \
  --private-data /private/private_assets/private_data \
  --selection /private/debug/train_fit_samples/selection.json \
  --max-samples 5 --output-dir /private/ops_fit_smoke_001
```

`selection.json` có `samples: [{"id": "..."}, ...]`; chỉ đọc ID. Kiểm SHA256
TRAIN JSONL, catalog, toàn bộ Python SDK và private split/index theo manifest
baseline50 trong repo. Kiểm selected IDs unique, giới hạn số mẫu, FIT membership,
audio nằm trong split, không thiếu/truncated, PCM16 mono 16kHz, environment tồn tại.
Không đọc/parse state. Audio dài vẫn giữ nguyên; không cắt audio để vượt budget.

Output directory phải mới, gồm:

- `prepared_audio.jsonl`: chỉ id/context/history/latest_audio. User history chỉ
  audio; assistant history giữ text/calls/results đã có, không replay.
- `expected_ids.jsonl`: ID theo selection, để kiểm coverage smoke; không thay
  expected IDs full public/VAL khi đánh giá cả split.
- `preflight.json`: hashes nguồn/input, audio durations, scope và giới hạn.

Không xuất label/current gold transcript, future tool results hoặc state vào
model input. Chuẩn bị đạt không xác nhận weights/runtime hoặc model quality:
`model_assets_verified=false`, `model_inference_run=false`. Mọi file chứa input
BTC phải để private local/Drive nhóm, không commit lên Git.

## Tiếp tục trên Windows CUDA

1. Giải nén model/private-data và runtime qua `llm/baseline_50_2/tools/unpack_assets.py`.
   Đối chiếu bộ ZIP với SHA256SUMS; file `...MODELS...-001.zip` đã kiểm trùng byte
   bản gốc, nhưng công cụ checksum tra theo tên archive nên cần đúng tên khi dùng.
2. Dùng môi trường inference theo lockfile baseline, không thay Python hệ thống.
3. Tạo config theo bốn path `assets_dir`, `dataset_dir`, `output_dir`,
   `llama_server`; dùng output_dir mới. Assets_dir có models/private_data/runtime.
4. Đặt env và chạy bounded audio smoke (không mở CLI public/VAL để debug):

```powershell
$env:LIFEMATE_BASELINE50_CONFIG = 'C:/private/config.local.json'
python ops/scripts/ops/run_system.py --factory lifemate.adapters.baseline50_factory:build_system --input C:/private/ops_fit_smoke_001/prepared_audio.jsonl --max-samples 5 --output-dir C:/private/ops_audio_run_001
python ops/scripts/ops/check_submission.py C:/private/ops_audio_run_001/submission.jsonl --expected-ids C:/private/ops_fit_smoke_001/expected_ids.jsonl
```

Factory kiểm đầy đủ assets/checksum rồi mới load model/server. Runtime hiện pin
Windows CUDA b11800; không chạy được exe đó trên macOS arm64. Linux/A30 cần một
variant runtime được kiểm chứng riêng, không bỏ SHA check để thay binary.

## Bằng chứng local 10/11

- Bộ ZIP assets baseline50 đã kiểm riêng: đủ 5 gói, checksum khớp, 180 file nội bộ
  đọc kiểm; các file pin bởi baseline manifest đầy đủ.
- Chuẩn bị 5 mẫu TRAIN-fit / 7 audio từ selection LLM bàn giao.
- Kiểm tích hợp oracle local: ASR và transport **giả lập bằng gold**, policy và
  SDK thật; đúng 5 session/5 mẫu, 5 batch ASR giả, calls/hash khớp references 5/5.
  Không gọi đây là model đạt 100%, không dùng thời gian fixture làm benchmark.
- Chưa chạy model/GPU thật; chưa chứng nhận latency toàn tuyến hoặc Docker offline.

Xem thêm [bridge](BASELINE50_BRIDGE.md). Suite unit dùng fixture nhỏ tự tạo,
không cần weights, audio BTC hoặc tài nguyên riêng tư trong CI.
