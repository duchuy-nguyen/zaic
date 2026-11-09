# Bridge baseline50 → Ops

Bridge trong `lifemate.adapters.baseline50` dùng trực tiếp code actor đã đóng
băng ở `llm/baseline_50_2/baseline50/actor.py`. Prompt, number/time hints, TF-IDF
example selection, native parser, repair text, call IDs và cách serialize
observation được giữ theo baseline50. Không sửa source LLM hoặc thêm scaffold
retrieval/grounding vào prompt của baseline.

## Đã kiểm và giới hạn

- 17 test bridge dùng fixture transport và source ToolEnv synthetic, so **từng
  request messages/tools/deadline/output reserve** với actor gốc trên các nhánh
  lookup→write→finish, ASK, malformed/schema repair, multi-call và budgets.
- So calls/hash từ session và trạng thái dừng. Kiểm missing/late ASR, late model,
  unknown execution không retry, split/example membership, policy isolation và
  cleanup context manager khi thành công/lỗi.
- 99 tests Ops pass trên Python 3.12 local; target workflow vẫn Python 3.11.
- Demo bridge 2 sample hoàn tất, failures=0. Chưa chạy weights/GPU thật; fixture
  parity không chứng minh output hay score 50,2 trên máy khác.

## Chạy demo ngay, không GPU/weights

Từ gốc checkout đủ `ops/`, `llm/` và `data/`:

```sh
python ops/scripts/ops/run_system.py \
  --factory lifemate.dev.baseline50_demo:build_system \
  --input ops/configs/ops/system_demo_input.jsonl \
  --max-samples 2 --output-dir ops/runs/baseline50_fixture_001
```

Output mới có manifest, traces, diagnostics và submission **fixture**. Hash là
hash fixture, không phải official ToolEnv; không nộp output này.

## Các thành phần

### Baseline50Policy(private_data)

Đọc `retrieval_index.json` và `split.json` của **baseline50**, kiểm bank chỉ chứa
source TRAIN-fit, FIT/VAL không giao nhau và group IDs khớp. Không thay bằng split
DATA 1237/308: baseline đã đóng băng FIT1236/VAL309.

Actor gốc dùng module globals cho retrieval. Bridge tạo namespace riêng cho
functions bằng FunctionType, dùng nguyên code/defaults của actor, chỉ bind
retriever/groups riêng và monotonic clock. Không rebind globals của baseline
gốc; hai policy instances không làm thay đổi bank của nhau. SHA256 source LLM,
split và index đi vào manifest. Lớp policy tự nó chưa xác minh weights/assets;
factory thật bên dưới dùng full verifier của baseline.

### Baseline50ASR(asr)

`transcribe(audio, remaining_seconds=...) -> str` hỗ trợ protocol một audio của
Ops. Với toàn sample, `prepare(sample, deadline=...)` gom **current audio trước,
sau đó các user-history audio đúng thứ tự**, gọi một lần
`MediumASR.transcribe(paths, deadline)` để giữ batch policy của baseline. Thứ tự,
normalization và prepared view được kiểm bằng `baseline50.asr.audio_view`.

Mapping phải đủ đúng references, text không rỗng và trả trước deadline. Không
fallback audio_text/gold khi lỗi. Prepared text được ưu tiên nếu có; mixed/text
input là chế độ riêng, không được gọi là baseline audio inference đã chứng nhận.
Trace Ops không xuất raw transcript/details hoặc raw model response.

### Baseline50System

Tương thích Ops batch CLI qua interface của LifeMateSystem, nhưng execution
engine là frozen baseline actor, **không phải Runner/PromptAugmenter** của
scaffold. environment_factory nhận ID-only Sample và trả một **raw SDK session**
có validate/call/get_tool_calls/get_state_hash; không trả ToolEnvAdapter ở bridge này.
SDK validation errors được giữ nguyên để repair text parity. Nếu prepared tool
allowlist hẹp hơn SDK, bridge bổ sung guard trước execute.

Một session/sample, không mở session thay thế khi lỗi; calls/hash luôn export
từ cùng session. Factory/export lỗi làm batch dừng, không bịa hash. Unknown
execution giữ call log thực và dừng, không retry.

Budget là cooperative, gồm environment creation, ASR, actor và session export;
không gồm model startup hoặc ghi trace/artifact cuối sample. Context=16384,
output=512 được giữ. `RuntimeConfig.max_steps` không được frozen actor sử dụng;
call/repair budgets vẫn có hiệu lực. Status `call_budget_exhausted` giữ đúng
baseline: không gắn failure hệ thống nhưng không coi là task success.

## Factory chạy thật: TRAIN-fit smoke

`lifemate.adapters.baseline50_factory:build_system` là context manager:

1. Đọc bốn-path config của baseline từ `LIFEMATE_BASELINE50_CONFIG`.
2. Dùng verifier baseline kiểm toàn bộ assets/checksum, TRAIN JSONL/catalog và
   Python runtime SDK. Không download hay chạy setup installer.
3. Nạp bank/split riêng, import đúng TRAIN SDK trong process mới.
4. Khởi tạo MediumASR và một LocalServer cho batch; đóng server khi thoát/lỗi.
5. Chỉ nhận sample ID thuộc **baseline TRAIN-fit**. Không dùng VAL/public để debug.

Ví dụ PowerShell, sau khi assets và môi trường Windows CUDA đã được bàn giao:

```powershell
$env:LIFEMATE_BASELINE50_CONFIG = 'C:/private/baseline/config.local.json'
python ops/scripts/ops/run_system.py --factory lifemate.adapters.baseline50_factory:build_system --input C:/private/prepared_fit_smoke.jsonl --max-samples 12 --output-dir ops/runs/baseline50_fit_smoke_001
```

Linux shell có thể đặt cùng env var, nhưng verifier hiện pin **Windows b11800
binary** theo baseline. Linux/Kaggle/A30 cần runtime variant được xác minh riêng;
bridge không bỏ kiểm SHA để chạy executable khác.

Input là prepared JSONL của Ops: id/context/history và latest_audio hoặc
latest_text. User-history dùng audio/text; không chuyển nguyên dòng train có
label/audio_text vào CLI. Với audio, references lấy đúng dataset row. Với text,
caller khai báo provenance text-oracle/actual-ASR phù hợp. Không có loader tự
chọn mẫu, không tự chạy lại public hoặc full VAL.

`--max-samples` kiểm số lượng trước khi load model. Output directory phải mới;
server logs nằm tại `<config.output_dir>/ops-train-smoke-server`, cũng phải mới.
Factory không warmup trên sample bổ sung. Lần chạy thật đầu tiên cần đo cold
inference và deadline; chưa xác nhận target latency.

CLI hỗ trợ cả factory trả LifeMateSystem và context manager yielding system.
Factory context được đóng khi batch hoàn tất hoặc exception. Không có resume
tự động ở CLI Ops này; giữ outputs lỗi để điều tra. Chính sách checkpoint/resume
của CLI baseline gốc không được thay thế hay suy là đã parity với batch Ops.

## Chạy tests

```sh
PYTHONPATH=ops/src:llm python -m unittest discover -s ops/tests/ops -v
```

Chưa xác nhận factory với weights/official audio thật. Cần các ZIP model/private
data, runtime và checksums do LLM bàn giao; official dataset/SDK đúng split; rồi
chạy bounded TRAIN-fit smoke. Không đổi model/bank/split chỉ để vượt preflight.
