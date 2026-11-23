# Hướng dẫn RTX 4060 → Test Solution BTC

Luồng nhóm đã chốt: dùng RTX4060 chạy thử Docker/ASR/LLM trên5 mẫu TRAIN-fit,
kiểm đầu ra và notebook, rồi export/upload để gửi Test Solution BTC. Không chờ
thuê/mượn RTX5090. Thời gian4060 chỉ là số đo local, không dự đoán thời gian5090.
Hướng dẫn này dành cho thành viên chạy máyGPU; chỉ chuẩn bị file/link để người
phụ trách đội gửi portal, không tự dùng quota khi chưa thống nhất.

Bản cập nhật12/11/2026. Mốc Test Solution:23:59 ngày15/11/2026 GMT+7.

## 1. Máy và tài nguyên cần có

- Linux x86_64, GPU NVIDIA và driver hỗ trợ CUDA13.2 hoặc tương thích với image.
  Dùng đúng RTX4060 hiện có. RAM máy16GB không có nghĩa VRAM16GB;
  lấy VRAM thực tế từ nvidia-smi. Không hứa model fit trước khi chạy.
- Ubuntu/Linux hoặc Ubuntu trong WSL2 trên Windows. Toàn bộ lệnh bên dưới chạy
  trong shell Linux/WSL, không dùng PowerShell. Windows dùng NVIDIA driver ở
  Windows và Docker Desktop GPU/WSL2 integration; không tự cài thêm Linux display
  driver trong WSL. Nếu chưa có DockerGPU, hoàn thiện bước đó trước.
- Docker Engine và NVIDIA Container Toolkit hoặc Docker Desktop WSL2 GPU integration đã sẵn sàng. Dự trù dung lượng cho image,
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

### Kiểm Docker nhận GPU trước khi build model

```sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
docker run --rm --gpus all nvidia/cuda:13.2.0-runtime-ubuntu24.04 nvidia-smi
python3 --version
docker info --format '{{.OSType}}/{{.Architecture}}'
```

Cần thấy RTX4060 bên trong container, Linux/x86_64 và Python>=3.11 trên host.
Bước này được tải base image khi setup, không phải offline inference. Nếu lỗi
`could not select device driver`, `unsatisfied condition cuda` hoặc không thấy
GPU, lưu log và xử lý driver/Docker trước. Không tự hạ CUDA/model để lách kiểm.

**Files cần nhận:** model ZIP, debug ZIP và train+public_test.zip đều đã có ở
nhóm. Debug có thể nằm bên trong Weights...002.zip: mở wrapper trước để lấy
DEBUG_AND_DATA_SAMPLES.zip. Không cần tải lại toàn bộ checkpoint train hoặc
runtime Windows để chạy Linux inference.

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

Nếu chỉ dựng candidate để thử kỹ thuật và chưa có training handoff hoàn chỉnh,
có thể chuẩn bị rõ ràng bộ source lịch sử như sau (không đổi nhãn thành đã reproduce):

```sh
mkdir -p "$TASKROOT/training_handoff"
cp -R llm/baseline_50_2/training_reference/. "$TASKROOT/training_handoff/"
cp llm/baseline_50_2/training_recipe.json "$TASKROOT/training_handoff/"
```

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

### Kiểm PyTorch/CUDA sau build

```sh
docker run --rm --gpus all --network none zaic-team:candidate python3 -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)'
```

Lỗi ở đây phải sửa trước smoke. Ghi lại cả log và image tag, không chỉ ảnh GPU.

## 5. Chạy smoke offline

Có thể mở terminal khác để ghi VRAM trong lúc smoke:

```sh
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv -l 1
```

Lưu output hoặc redirect vào gpu-monitor.csv, dừng bằng Ctrl+C sau lượt chạy.
Đây là memory.used toàn GPU (có thể gồm app khác), không phải riêng allocator
của model. Đóng bớt ứng dụngGPU bằng thao tác bình thường trước khi thử.

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

### Cách quyết định sau smoke trên4060

| Kết quả | Làm tiếp |
|---|---|
| OOM, server không khởi động, thiếu CUDA/lib, output rỗng/thiếu | Chưa export để gửi test. Giữ log và nhờ Ops sửa candidate. |
| Có5 output nhưng diagnostics đều asr_error/backend_error hoặc không có action | Chưa coi là model chạy thành công; kiểm log. Một số mẫu calls=[] là đúng nhưng không chứng minh cảpipeline đã chạy. |
| Có đủ5 output, ASR/LLM thực sự chạy, calls/hash đúng nguồn và không lỗi kỹ thuật | Kiểm notebook rồi chuẩn bị Test Solution. |
| Chỉ có mẫu>30s trên4060 | Ghi rõ, phân tích; không kết luận5090 sẽ fail và không tăng budget30s. Có thể ưu tiên test khả năng chạy trên BTC sau khi xác nhận image/output hợp lệ. |
| Điểm/chọn tool sai nhưng pipeline hoạt động | Báo quality lỗi riêng; Test Solution không chấm điểm, không cần đạt ngưỡng accuracy tự đặt để thử. |

Nếu OOM: gửi tênGPU/VRAM, free VRAM lúc bắt đầu, server.log và traceback. Không tự
đổi weights, context16384, ASR batch, quantization hoặc GPU layers rồi gọi đó là
baseline đã chốt. Ops/LLM sẽ version một cấu hình/variant riêng nếu cần.

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

## 7. Gửi bằng chứng và chuẩn bị image Test Solution

Gửi riêng cho nhóm: build.log, predict.log, gpu.txt, image.json, pip-freeze.txt,
hai submission, diagnostics, server logs và notebook output/errors. Cho biết
commit repo đã dùng và phần nào chưa chạy được. Không push data/weights/logs thô lên Git.

Sau khi Ops/LLM xác nhận đủ điều kiện **thử kỹ thuật**, export đúng image đã
chạy. Không yêu cầu có5090 hoặc phải xong fullVAL. Yêu cầu training/reproduce
cuối phải được theo dõi riêng: test-run thành công không chứng nhận phần đó.
Image gửi chính thức cuối vẫn phải hoàn thiện Training Code/Data đúng guide.

Dùng tag nhận diện bất biến trước khi save để tránh nhầm image đang sửa:

```sh
docker image inspect zaic-team:candidate --format '{{.Id}}'
docker tag zaic-team:candidate zaic-team:rtx4060-test1
```

Sau đó:

```sh
docker save zaic-team:rtx4060-test1 | gzip > "$TASKROOT/zaic_team.tar.gz"
md5sum "$TASKROOT/zaic_team.tar.gz" > "$TASKROOT/zaic_team.tar.gz.md5"
```

Sau export, kiểm archive load được bằng `docker load -i "$TASKROOT/zaic_team.tar.gz"`.
Nếu có máyLinux khác thì rehearsal thêm; không cần có5090 mới được gửi Test Solution. Sau đó người phụ trách
mới upload archive lên Drive, bật quyền tải bằng link và gửi MD5 lên Test Solution.
Không upload image I/O hoặc image chưa có model. Test Solution đóng23:59 ngày
15/11/2026 GMT+7, tối đa1lượt/ngày/đội; không cần đợi full VAL mới thử Docker
nhưng phải bảo đảm image chạy thật và đầu ra đúng contract.


## 8. Upload Drive và gửi Test Solution

1. Upload **zaic_team.tar.gz** lên Google Drive của nhóm. Bật Anyone with the link
   cho đúng filearchive để BTC tải được; không dùng link folder thay file.
2. Copy MD5 ở zaic_team.tar.gz.md5. MD5 tính trên archive đã gzip, không dùng
   SHA256 weights, image ID hoặc checksum của file trước khi nén.
3. Đội trưởng/người phụ trách mở Portal → LifeMate → **Test Solution**.
4. Nhập link Google URL và MD5, kiểm lần cuối rồi Submit Test.
5. Chờ hệ thống xong. Nếu lỗi, chọn View full log và gửi toàn bộ log/trạng thái
   về Ops. Nếu thành công, lưu ảnh/trạng thái và ngày giờ; có thể không có log
   chi tiết theo thông báo BTC.

Mỗi đội tối đa1test/ngày. Tối đa5thao tác lại chỉ cho lỗi link/checksum/tảifile,
không suy thành5 lượt chạy model. Test Solution không tính lượt nộp chính thức,
không chấm score và không tự xác nhận toàn bộ test đều dưới30s.

Khi sửa image: đổi tag, export archive mới, tính lại MD5 và upload đúng bản mới.
Không đổi filearchive/MD5 trong khi BTC đang tải/chạy. Sau khi test thành công,
đánh dấu chính xác imageID/gitcommit đã được thử; sửa code sau đó là bản khác,
không dùng trạng thái pass cũ để chứng nhận image mới.

### Tin nhắn gửi lại nhóm

> RTX4060 [VRAM thật], driver [...], commit [...], imageID [...]. Build [đạt/lỗi],
> smoke5 mẫu [đủ/thiếu], notebook [đạt/lỗi], số mẫu>30s [...], peakVRAM/cách đo [...].
> Archive/link [...], MD5 [...]. Test Solution [chưa gửi/đang chạy/pass/fail].
> Đính kèm build.log, predict.log, server logs, diagnostics và hai file JSONL.

Nguồn contract: Docker_Guide_ZAC2026_LifeMate.pdf của BTC; thông báo Test Solution
ngày10/11. Không phụ thuộc việc đội có RTX5090 để bắt đầu quy trình này.
