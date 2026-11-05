# Rà soát bàn giao và tích hợp Ops — 05/11/2026

Đối chiếu main tại `920a7a3`; fetch không có commit mới hơn. Các nhánh DATA,
LLM, Speech, Ops hiện không có commit riêng chưa nằm trong main.

## Đã nhận qua Git

| Phía | Có thể dùng | Chưa có trong repo clone sạch |
| --- | --- | --- |
| DATA (`e9db85f`, `44467ad`, `ebdf99d`, merge `a086a9d`) | Pipeline T1–T4; 488 records synthetic, meta, catalog 147 tools; runtime `data/synthetic/zaic_tools`; train-fit provisional IDs | Official dataset/audio; train/public runtime nguyên bản riêng theo split; `synthetic_environment/<id>/state.json`; manifest nguồn synthetic được report nhắc đến |
| Speech (PR #5, merge `a72bf39`) | ASR/cache/metrics/loader, cấu hình revision PhoWhisper, định dạng transcript và tests | Weights, transcript/audio và run artifacts riêng; chưa có bằng chứng target GPU từ repo |
| LLM (PR #3, merge `d827fff`) | `initial_messages`, `allowed_schemas`, `parse_decision`; transport `LocalServer.generate`; agent/scorer và CPU tests | Weights, installation manifest/runtime llama.cpp, split-v1/expanded manifests và snapshots cần cho các full/expanded tests |
| Ops | Runner, budgets/repair, logs/manifest, config/submission structural checker | Hard cancellation, ASR end-to-end budget, offline packaging và timing A30 |

Báo cáo DATA nói FIT 1342 / VAL 203; hướng dẫn thí nghiệm LLM có nhắc VAL 309.
Không tự coi đây là cùng split. Cần DATA/LLM bàn giao manifest ID/hash/version và
chốt split chung trước so sánh chất lượng hoặc freeze B0–B3.

## Đã bắt đầu dùng trong Ops

- `src/lifemate/adapters/toolenv.py`: inject đúng ToolEnv constructor, root và split;
  tạo một session/sample. Ops không parse state.json, không gọi get_state trong
  đường model; validate trước execute; không tự retry execution không rõ trạng thái.
- `src/lifemate/adapters/native_llm.py`: dùng prompt/schema/parser của LLM;
  inject transport có interface `generate(messages, tools, deadline, max_tokens=...)`.
  Chuyển observations thành assistant/tool messages, repair feedback về lượt model;
  không tự launch server/download model hoặc thực thi tool từ backend.
- `tests/ops/test_adapters.py`: 11 integration tests dùng source runtime và catalog
  đã commit, state tự tạo trong thư mục tạm và generation fixture. Không lấy label
  synthetic làm model response. Kiểm lookup→update, isolation, ask-stop, parse/schema
  repair, multi-call rejection, done=false, transport error và unknown execution.

Kết quả local: **61 tests Ops pass** (50 tests cũ + 11 tests adapter).
Job Ops hiện chạy unittest discovery nên tự nhận tests mới, không cần đổi workflow.

`done` không xuất hiện trong một số observation của runtime (ví dụ ask). Adapter
để cờ nội bộ done=false khi thiếu và giữ nguyên payload trong Observation.data;
không tự diễn giải ok=true/finish là task success. Call/hash luôn từ session.
Hash runtime có thể loại free-text fields khi canonicalize; không dùng hash đổi/
không đổi làm bằng chứng duy nhất rằng nội dung văn bản đã được cập nhật.

Đây là **integration với source runtime đã nhận + state/model fixtures**.
Không phải official-data replay, model inference, B0A hoặc submission hợp lệ.
Không sửa source/catalog của DATA hoặc prompt/parser/transport của LLM.

## Cách ghép khi đủ tài nguyên riêng

Chạy Python với `ops/src`, `llm` và thư mục chứa **đúng runtime của split** trên
PYTHONPATH. Mỗi process chỉ load một bản `zaic_tools`; không import train rồi đổi
sys.path sang public_test trong cùng process. Không mặc định dùng bản synthetic
cho train/public_test.

```python
from pathlib import Path
from zaic_tools import ToolEnv  # đúng source runtime/split đã pin
from lifemate.adapters.toolenv import ToolEnvFactory
from lifemate.adapters.native_llm import NativeLLMBackend
from lifemate.runtime.runner import Runner
from lifemate.runtime.contracts import Sample

# catalog_by_name, prepared_input và server do DATA/LLM cung cấp.
# server đã khởi tạo; backend không sở hữu lifecycle server.
factory = ToolEnvFactory(ToolEnv, root=Path('/private/data-root'), split='train')
backend = NativeLLMBackend(server, catalog_by_name)
result = Runner(factory).run(Sample(sample_id, prepared_input), backend)
```

`prepared_input` gồm latest_text, context và history đã chuẩn bị. TRAIN gold-text
chỉ dùng cho thí nghiệm được khai báo; public/private phải dùng transcript ASR,
không fallback gold. Không truyền label/state snapshot vào model input.
`session_output()` chỉ cung cấp calls/hash cho bên sở hữu session; chưa có CLI
submission writer hoặc chính sách chấp nhận kết quả failure/unknown.

## Bàn giao tiếp theo cần ưu tiên

1. DATA: vài sample TRAIN và environment tương ứng, train catalog/runtime riêng,
   root layout, archive/runtime/catalog hashes; manifest split và ID đã chốt.
2. LLM: resource root/installation manifest/weights có sẵn, lệnh server và một
   smoke response thật; xác nhận adapter repair messages và generation contract.
3. Speech: một transcript artifact current/history có key/coverage, provenance và
   lỗi rõ ràng để nối input; không tải model hoặc chạy lại full ASR chỉ để test Ops.
4. Ops: chạy text smoke trên train thật; sau đó nối ASR, đo toàn pipeline, review
   provenance và packaging. Chưa đo latency GPU hoặc xác nhận task quality.
