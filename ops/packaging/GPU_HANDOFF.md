# Hướng dẫn thành viên chạy thử Docker LifeMate trên GPU

Mục tiêu: build candidate từ `zaic1/main`, chạy ASR/LLM thật trên 5 mẫu TRAIN-fit,
kiểm notebook và gửi lại bằng chứng cho Ops. Chưa tự gửi Test Solution.

## 1. Máy và tài nguyên cần có

- Linux x86_64, GPU NVIDIA và driver hỗ trợ CUDA13.2 hoặc tương thích với image.
  Ưu tiên RTX5090 32GB như hướng dẫn BTC; GPU khác chỉ chứng minh smoke trên máy đó.
- Docker Engine và NVIDIA Container Toolkit đã cài. Dự trù dung lượng cho image,
  weights4.2GB, build context và archive export; không build nếu disk gần đầy.
- Python3.11+ trên host, git, curl; có quyền đọc repo private `zaic1`.
- Bộ LLM bàn giao: MODELS_AND_PRIVATE_DATA ZIP, DEBUG_AND_DATA_SAMPLES ZIP;
  official dataset BTC `train+public_test/`. Không cần runtime Windows.
- Training handoff riêng: code/data hoặc URLs bất biến và README tái lập.
  Nếu mới smoke kỹ thuật, được dùng source lịch sử để tạo candidate và phải ghi
  rõ chưa kiểm reproduction; không coi đó là bộ training cuối để nộp.

Kiểm trước:

```sh
uname -m
nvidia-smi
docker version
df -h
```

Chạy các bước trong cùng một shell. Quy ước thư mục dưới đây có thể đổi, nhưng
giữ mọi weights/data ngoài checkout Git:

```sh
TASKROOT="$HOME/lifemate-handoff"
mkdir -p "$TASKROOT/incoming"
cd "$TASKROOT"
git clone https://github.com/duongtanminh1912/zaic1.git repo
cd repo
git switch main
git pull --ff-only
```

Nếu đã clone, chỉ `cd` vào repo rồi pull main. Không dùng repo cũ duchuy-nguyen.
Chép các ZIP LLM vào `$TASKROOT/incoming/`; giải nén bộ BTC thành
`$TASKROOT/train+public_test/` (bên trong có train/ và public_test/).

## 2. Giải nén model và stage runtime Linux

Dùng `python3` phiên bản >=3.11:

```sh
python3 llm/baseline_50_2/tools/unpack_assets.py \
  --archive "$TASKROOT/incoming/LIFEMATE_BASELINE_50_2_MODELS_AND_PRIVATE_DATA-001.zip" \
  --dest "$TASKROOT/private_assets"

curl -fL --retry 2 \
  https://github.com/ggml-org/llama.cpp/releases/download/b11800/llama-b11800-bin-ubuntu-cuda-12.8-x64.tar.gz \
  -o "$TASKROOT/incoming/llama-linux-cuda.tar.gz"

python3 ops/packaging/stage_linux_runtime.py \
  "$TASKROOT/incoming/llama-linux-cuda.tar.gz" "$TASKROOT/linux_runtime"
```

Đổi tên archive model trong lệnh nếu file thực tế mang tên không có `-001`.
Helper runtime tự đối chiếu SHA upstream; không thay bằng bản latest.
Giải nén DEBUG_AND_DATA_SAMPLES ZIP vào `$TASKROOT/debug/`, sao cho có
`$TASKROOT/debug/train_fit_samples/selection.json`.

## 3. Chuẩn bị đúng 5 mẫu smoke (không chạy cả TRAIN)

```sh
python3 ops/scripts/ops/prepare_baseline50_smoke.py \
  --dataset-dir "$TASKROOT/train+public_test" \
  --private-data "$TASKROOT/private_assets/private_data" \
  --selection "$TASKROOT/debug/train_fit_samples/selection.json" \
  --max-samples 5 --output-dir "$TASKROOT/smoke_inputs"
```

Phải báo 5 samples/7 audio. Tạo layout dữ liệu nhỏ để mount Docker. Đây là bước
chuẩn bị TRAIN ngoài inference: sao chép state nguyên byte, không đưa vào prompt.

```sh
export TASKROOT
python3 - <<'PY'
import json, os, pathlib, shutil
r = pathlib.Path(os.environ['TASKROOT'])
source = r/'train+public_test/train'
target = r/'smoke_data/train'
target.mkdir(parents=True, exist_ok=False)
shutil.copytree(source/'zaic_tools', target/'zaic_tools',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copy2(source/'train_tools_catalog.json', target/'train_tools_catalog.json')
prepared = [json.loads(line) for line in (r/'smoke_inputs/prepared_audio.jsonl').read_text().splitlines()]
with (target/'train.jsonl').open('x', encoding='utf-8') as output:
    for item in prepared:
        row = {'id': item['id'], 'audio': item['latest_audio'],
               'context': item['context'], 'history': item['history']}
        output.write(json.dumps(row, ensure_ascii=False)+'\n')
        refs = [row['audio']] + [t['audio'] for t in row['history'] if t['role']=='user']
        for ref in refs:
            path = target/ref
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/ref, path)
        relative = pathlib.Path('train_environment')/row['id']/'state.json'
        (target/relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target/relative)
print('SMOKE_DATA_READY', len(prepared))
PY
```

Không mount toàn bộ TRAIN rồi chạy mặc định vì sẽ thành1545 mẫu. Không dùng VAL
hoặc public để sửa lỗi tích hợp. Không sửa SDK/state hay tự tạo transcript gold.

## 4. Tạo build context và build

Đặt training handoff vào `$TASKROOT/training_handoff/`, có README.md. Để chỉ thử
candidate khi chưa nhận portable training, có thể copy `training_reference/`
và recipe từ baseline vào thư mục đó; README gốc ghi rõ giới hạn, không sửa thành
“đã tái lập”. Trước nộp cuối cần thay bằng bàn giao đã hoàn thiện từ LLM.

```sh
python3 ops/packaging/prepare_context.py \
  --assets "$TASKROOT/private_assets" \
  --runtime "$TASKROOT/linux_runtime" \
  --training "$TASKROOT/training_handoff" \
  --destination "$TASKROOT/docker_context"

set -o pipefail
docker build --platform linux/amd64 -t zaic-team:candidate \
  "$TASKROOT/docker_context" 2>&1 | tee "$TASKROOT/build.log"
```

Nếu thiếu CUDA libraries hoặc dependency, giữ nguyên log và gửi Ops; không đổi
model/prompt/budget để “chạy cho qua”. Dockerfile/requirements là candidate,
chưa được xác nhận trên GPU. Build có Internet; chạy inference phải tắt mạng.

## 5. Chạy smoke offline

```sh
mkdir -p "$TASKROOT/result_smoke"
docker run --rm --gpus '"device=0"' --network none \
  -v "$TASKROOT/smoke_data:/data:ro" \
  -v "$TASKROOT/result_smoke:/result" \
  zaic-team:candidate /bin/bash /code/predict.sh \
  2>&1 | tee "$TASKROOT/predict.log"
```

Kết quả cần có:

- `result_smoke/submission.jsonl`: đủ5 ID, mỗi dòng id/tool_calls/state_hash/time.
- `submission.jsonl.diagnostics.jsonl`: status, failure, thời gian và provenance.
- stdout có model_load_ms, predict_ms; không có lỗi thiếu library/CUDA/ASR/model.
- Calls/hash là output session thật. Không yêu cầu cả5 mẫu đúng gold để chứng
  minh Docker chạy được; vẫn phải báo mọi mẫu sai hoặc timeout, không loại khỏi báo cáo.

Nếu dừng giữa chừng, giữ thư mục result/log. Không xóa output rồi tự chạy lại để
chọn kết quả. Báo Ops xác định nguyên nhân trước khi mở lượt smoke mới.

## 6. Notebook và thông tin môi trường

```sh
docker run --rm --gpus '"device=0"' -p 127.0.0.1:9777:9777 \
  -v "$TASKROOT/smoke_data:/data:ro" \
  -v "$TASKROOT/result_smoke:/result" \
  zaic-team:candidate /bin/bash /code/start_jupyter.sh
```

Mở localhost:9777, token `zac2026`, mở predict_notebook.ipynb, dùng kernel mới
và Run All. Notebook phải xuất `jupyter_submission.jsonl`, không overwrite
submission.jsonl. So cùng ID/calls/hash; thời gian có thể khác. Việc hai lượt
khác kết quả phải được ghi nhận để điều tra, không sửa tay cho giống nhau.

Ghi lại môi trường:

```sh
nvidia-smi > "$TASKROOT/gpu.txt"
docker image inspect zaic-team:candidate > "$TASKROOT/image.json"
docker run --rm --network none zaic-team:candidate python3 -m pip freeze > "$TASKROOT/pip-freeze.txt"
docker run --rm --network none zaic-team:candidate python3 -m pip check
```

Ghi peak VRAM bằng công cụ theo dõi GPU của máy trong lúc chạy; gửi cách đo,
không lấy VRAM nhàn rỗi hoặc nvidia-smi trước inference làm peak. Kiểm giới hạn
30giây/mẫu bằng `time`/log; smoke GPU khác RTX5090 chưa chứng nhận máy chấm.

## 7. Gửi lại Ops và chỉ export sau khi chốt

Gửi riêng cho nhóm: build.log, predict.log, gpu.txt, image.json, pip-freeze.txt,
hai submission, diagnostics, server logs và notebook output/errors. Cho biết
commit repo đã dùng và phần nào chưa chạy được. Không push data/weights/logs thô lên Git.

Sau khi Ops/LLM xác nhận candidate đủ điều kiện và training handoff đã hoàn thiện:

```sh
docker save zaic-team:candidate | gzip > "$TASKROOT/zaic_team.tar.gz"
md5sum "$TASKROOT/zaic_team.tar.gz" > "$TASKROOT/zaic_team.tar.gz.md5"
```

Thử docker load/chạy offline trên máy khác trước bàn giao. Sau đó người phụ trách
mới upload archive lên Drive, bật quyền tải bằng link và gửi MD5 lên Test Solution.
Không upload image I/O hoặc image chưa có model. Test Solution đóng23:59 ngày
15/11/2026 GMT+7, tối đa1lượt/ngày/đội; không cần đợi full VAL mới thử Docker
nhưng phải bảo đảm image chạy thật và đầu ra đúng contract.
