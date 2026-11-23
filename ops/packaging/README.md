# Docker candidate — LifeMate, 11/11/2026

Đây là component đóng gói riêng trong `ops/packaging/`, dùng lại inference/policy
baseline50 qua import, không sửa frozen source. Helper I/O ở
`ops/src/lifemate/runtime/container_io.py`; tests ở `ops/tests/ops/test_container_io.py`.
Build context chứa tài nguyên riêng tư được tạo ngoài Git checkout.

**Chưa phải image cuối để gửi Test Solution.** Đã kiểm I/O/fixture trong Docker
Linux AMD64 không mạng, chưa chạy weights trên NVIDIA GPU hoặc gửi portal.

Nguồn: [Docker Guide BTC](https://dl-challenge.zalo.ai/2026/lifemate_argfzdgert/Docker_Guide_ZAC2026_LifeMate.pdf),
đã đọc 19 trang. Máy chấm mới: RTX5090 32GB, Xeon Gold6459C 16 core, RAM64GB,
driver595.58.03, CUDA13.2. Các mục A30 trong plan cũ là lịch sử. Trang4 cho phép
CUDA<=13.2 nhưng trang7 ghi bắt buộc13.2; candidate chọn base13.2, đã kiểm registry
và pin digest AMD64. Không coi runtime Windows là Linux variant đã kiểm chứng.

## Contract mới

- `/code/predict.sh` nhận `/data/<split>` và ghi `/result/submission.jsonl`.
- SDK import từ `/data/<split>/zaic_tools`, không đóng TRAIN SDK thay private.
- Output thêm `time` integer milliseconds, bên cạnh id/tool_calls/state_hash,
  theo PDF trang7/17. Checker riêng không đổi format baseline lịch sử.
- Có JupyterLab, start_jupyter.sh và predict_notebook.ipynb; cùng engine,
  ghi `/result/jupyter_submission.jsonl` và giữ output predict cũ. BTC chỉ dùng
  notebook timing khi điểm hai output giống nhau.
- Model/tokenizer nằm trong image; inference offline; một ToolEnv/mẫu và không
  đọc state/get_state, không replay history hoặc tự reset khi lỗi.
- stdout có model_load_ms và predict_ms. Timer từ trước đọc/prepare từng input,
  qua ASR current/history, tools tới SDK result; chưa gồm flush output, như ví dụ
  BTC. Cooperative deadline không phải hard cancellation. Cần đo GPU thực tế.
- Phải có training_code/ và README phương pháp, code/data hoặc URLs bất biến để
  tái lập train; không đưa base weights chỉ dùng training vào image.

## Kiểm I/O trên Mac (không GPU)

Từ repository root:

```sh
docker build --platform linux/amd64 -f ops/packaging/Dockerfile.io -t zaic-ops-io:local .
docker run --rm --platform linux/amd64 --network none -v /private/data:/data:ro zaic-ops-io:local --check-input --split public_test
```

Image này chỉ stdlib/source, không gửi BTC. Preflight không load model, không
tạo ToolEnv hoặc predictions. Đã đọc đủ634 public inputs với network none,
data read-only. 137 Ops tests pass trong Python3.11 LinuxAMD64 Docker emulation;
không suy ra tốc độ GPU từ test đó.

## Tạo context riêng tư và build candidate

Tải [llama.cpp b11800 Linux CUDA12.8](https://github.com/ggml-org/llama.cpp/releases/tag/b11800).
Archive SHA256 pin: e5d489b995c2a0965ee746a246c934be9f4ffa00c78183b2d4f644f43ae5d7a7.
Đã tải/check/stage62 file, parity_verified=false. Không thay verifier Windows.

```sh
python3 ops/packaging/stage_linux_runtime.py /private/llama-b11800-bin-ubuntu-cuda-12.8-x64.tar.gz /private/linux_runtime
python3 ops/packaging/prepare_context.py --assets /private/private_assets --runtime /private/linux_runtime --training /private/training_handoff --destination /private/zaic_docker_context
docker build --platform linux/amd64 -t zaic-team:candidate /private/zaic_docker_context
```

Model ZIP được giải nén bằng tools/unpack_assets.py của baseline trước đó.
Helper kiểm model pin và runtime hashes, chỉ copy model/index/split từ assets;
không đưa VAL refs/public traces vào inference. Training handoff phải riêng.
Linux requirements là candidate, chưa lock GPU; sau smoke cần pip freeze/check,
driver, image digest và bằng chứng parity. CUDA12.8 libraries cho server lấy từ
torch NVIDIA wheels trên base13.2; LD path setup trong predict.py cần GPU smoke.

Mac còn khoảng17GiB free tại lần kiểm: full CUDA image/model/context/export có
thể không đủ dung lượng. Build/export cuối trên Linux GPU đủ chỗ; không tự prune
image/volume dự án khác.

## Rehearsal theo lệnh BTC

```sh
docker run --rm --gpus '"device=0"' --network none -v /private/test_data:/data:ro -v /private/result:/result zaic-team:candidate /bin/bash /code/predict.sh
docker run --rm --gpus '"device=0"' -p 127.0.0.1:9777:9777 -v /private/test_data:/data:ro -v /private/result:/result zaic-team:candidate /bin/bash /code/start_jupyter.sh
```

Token Jupyter mẫu zac2026 theo guide; chỉ publish localhost khi phát triển.
Notebook fresh kernel/run-all, startup một lần, không warm-up bằng public/VAL.
Kiểm output đủ ID, calls/hash, điểm predict/notebook, latency/VRAM, offline.
Output cũ được giữ, không auto retry; container candidate là luồng một lượt,
không áp dụng resume VAL/public cũ lên private/in-flight session chưa rõ.

## Những mục còn chặn image cuối

- [ ] Linux server/ASR/Qwen chạy thật trên RTX5090, kiểm executable/CUDA libs.
- [ ] Đo input→result30s/sample, VRAM/context và predict/notebook parity.
- [ ] Build/load image và rehearsal offline trên LinuxGPU, dependency lock thật.
- [ ] Training Code/Data tái lập: training_reference hiện phụ thuộc layout/kernel
  lịch sử, không standalone. Cần LLM bàn giao entrypoint prepare/train/merge/
  quantize, base revisions, môi trường và data indexes/URL bất biến.
- [ ] Kiểm license/provenance và README giải pháp thực dùng.
- [ ] docker save tar.gz, MD5, Drive và Test Solution sau yêu cầu gửi của user.

Test Solution theo email BTC10/11: đóng23:59 GMT+7 ngày15/11,1lượt/ngày, không
chấm điểm. Email mới hơn FAQ09/11 có dòng cũ nói chưa test-run. Không upload
image io-test/candidate thiếu model. Không có gửi thư/submit/upload trong task này.

FAQ mới xác nhận optional args hợp lệ có thể match gold, ask params so như tập.
Scorer local-spec-v1 cũ dùng exact key set/list order: giữ để so lịch sử nhưng
không gọi57.65 là official score đã cập nhật theo FAQ.
