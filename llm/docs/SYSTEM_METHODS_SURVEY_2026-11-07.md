# LifeMate: survey phương pháp ngoài fine-tuning/distillation

Ngày: 07/11/2026. Code đối chiếu: `4f09426`. Phạm vi: thiết kế hệ thống và kế hoạch đo; chưa triển khai các module đề xuất, chưa chạy training hoặc benchmark mới.

## 1. Quyết định đề xuất

Xây hệ thống theo hướng **ASR + agent nhỏ + harness có kiểm soát + retrieval có chọn lọc + grounding**. Fine-tuning/distillation là một trục cải thiện model; các trục inference, dữ liệu và vận hành phải được đo riêng.

Ưu tiên gần hạn: harness/evaluation → grounding → constrained decoding → tool retrieval → retrieval ví dụ train → speech/context optimization. Không thay teacher/student hoặc coi benchmark ngoài là bằng chứng tăng điểm LifeMate.

**Phân loại bằng chứng:**

- **FACT/CODE:** chỉ các hành vi đã đọc trong source ở revision trên; không đồng nghĩa đã chạy full pipeline trên Kaggle/A30.
- **RESEARCH:** kết quả hoặc cơ chế trong nguồn sơ cấp; không suy rộng thành mức tăng điểm LifeMate.
- **TEAM OBSERVATION:** các lỗi OOM/timeout/format và metrics được cung cấp trước đây; chưa có benchmark mới trong survey này.
- **PROPOSAL/HYPOTHESIS:** kiến trúc, tham số và experiment bên dưới; cần kiểm chứng trên frozen validation.
- **OPEN RULE:** portal LifeMate mở được nhưng không trả nội dung đọc được qua công cụ web trong lần khảo sát này. Chưa xác minh lại model cap, deadline, Docker contract hoặc quyền đóng gói retrieval corpus. Các giới hạn trong context của team vẫn cần README/FAQ BTC xác nhận. Không dùng mirror để tự giải quyết xung đột luật.

## 2. Đối chiếu hệ thống hiện tại

| Thành phần | FACT/CODE | Khoảng trống/giới hạn |
| --- | --- | --- |
| Notebook agent | `llm/notebooks/oneclick/core.py`: `schemas`, `run_agent`, `add_observation`; lọc installed tools, validate trước call, bounded repair, giữ observation | Hiện đưa toàn bộ schema được cài; chưa thấy tool retriever hoặc entity store riêng trong đường này |
| Submission notebook | Calls/hash lấy từ cùng env ở cuối `run_agent` | Giữ nguyên bất biến này khi thêm retrieval/guard; không xuất proposals thay execution log |
| Ops harness | `ops/src/lifemate/runtime/runner.py`: deadline, step/call/repair budgets, execution unknown, trace | Cần chứng minh tích hợp model + audio + official split; mock/test fixture chưa đủ |
| Ops adapters | `adapters/toolenv.py`, `adapters/native_llm.py` đã tồn tại | `runtime/README.md` còn nói chưa có bridge/parser, không phản ánh adapter hiện có; cần cập nhật khi tích hợp |
| Hai đường inference | Notebook dùng controller riêng; Ops native adapter tái sử dụng `local_baseline.agent` | Nguy cơ lệch prompt/parser/budgets/STOP giữa notebook evaluation và Docker. Cần contract test chung trước khi hợp nhất |
| Speech | `speech/src/lifemate_speech/entities.py` có weak-regex diagnostics; speech baseline pin PhoWhisper-small | Chưa phải semantic entity resolver. Notebook medium LoRA và speech-small là hai config khác nhau, không coi metrics thay thế cho nhau |
| Training notebook | Precheck, checkpoint/reuse, EOS fix, bounded attention/loss đã có | Full student 4B/Kaggle T4 và runtime A30 vẫn cần xác minh; survey không chứng nhận các bước này |

## 3. Survey và lựa chọn

### A. Agent harness: P0, nền tảng bắt buộc

Harness là lớp điều phối quanh model: chuẩn bị input, giới hạn tool, parse/validate, execute, cập nhật observation, dừng, trace và export. Chọn LangGraph hoặc viết Python trực tiếp là lựa chọn implementation; dùng framework không tự tăng độ chính xác.

**Đề xuất:** phát triển tiếp contracts/runner sẵn có; một next action mỗi lượt, explicit STOP và ASK. Lưu task ledger nhỏ: yêu cầu còn lại, facts đã biết, call/result đã thực thi, thông tin thiếu. Ledger là trạng thái suy luận nội bộ, không phải state thật của ToolEnv.

Guard kiểm tra allowlist/schema và provenance của ID. Chống lặp bằng phát hiện chuỗi không tiến triển; không chặn tuyệt đối mọi call giống nhau vì một số read hoặc thao tác lặp có thể hợp lệ. Write chưa rõ side effect phải dừng với log thật, không retry hoặc rollback giả.

**Kỳ vọng:** giảm call thừa, loop, lỗi schema và thiếu dependency. Đo extra/missing calls, invalid rate, budget stops, state match và latency. ReAct là nguồn tham khảo cho vòng action–observation; không cần sao chép reasoning dài vào output. [ReAct](https://arxiv.org/abs/2210.03629)

### B. Tool RAG / schema retrieval: P1

Index tên tool, description, required/enum/x-ref và mô tả returns từ catalog thực tế. Mỗi sample lọc installed tools trước; truy vấn bằng transcript, phần history liên quan và observation hiện tại.

**Baseline đề xuất:** lexical BM25 hoặc TF-IDF word/character n-gram trên CPU; không thêm embedding model ngay. So với full permitted catalog. Thử k=4/8/12 như cấu hình experiment, không xem là giá trị tối ưu.

Giữ `sys.ask_missing_info`; bổ sung các read/dependency tools liên quan theo schema đã kiểm chứng. Re-retrieve sau observation khi task đổi bước; có fallback mở rộng về toàn bộ permitted catalog khi retrieval thiếu chắc chắn. Điểm similarity thấp không đủ kết luận OOS.

**Rủi ro:** bỏ mất lookup tool hoặc tool của app thứ hai làm task không thể hoàn thành. Nếu installed catalog đã nhỏ, retrieval có thể không có lợi. Đo tool-set recall@k trên gold validation, prompt tokens, retrieval time và score cuối. Gold chỉ dùng để đánh giá recall, không dùng chọn tool lúc inference.

ToolLLM kết hợp tool-use model với API retriever; điều này hỗ trợ khảo sát cơ chế retrieval, không chứng minh BM25 hay k cụ thể tốt cho LifeMate. [ToolLLM](https://arxiv.org/abs/2307.16789)

### C. Example / trajectory retrieval: P2

Retrieval 1–3 ví dụ từ **train-fit** có intent, app/tool family và history pattern gần yêu cầu hiện tại. Ví dụ phải là cặp input–action hoặc trajectory có provenance và đã kiểm chứng; tách rõ demonstration với yêu cầu đang thực hiện.

Delexicalize ID, người nhận, địa chỉ, số tiền thành placeholders trong demonstration khi phù hợp. Không copy ID hoặc arguments từ sample khác; mọi binding phải được ground lại vào sample hiện tại. Không biến placeholder thành giá trị execute.

Build index sau khi freeze split; không có validation/public/private labels trong index. Train-fit audio inference dùng transcript ASR làm query, không dùng current gold transcript. Các ví dụ chứa dữ liệu BTC phải giữ private; quyền đóng gói corpus trong final image cần đối chiếu rule.

**Kỳ vọng:** giảm wrong tool, ask/stop sai và lỗi pattern hiếm mà không cập nhật weights. **Rủi ro:** nhiễu, copy entity, prompt dài. So 0 vs 1 vs 3 ví dụ, báo cả score và latency. Synchromesh là tham khảo về retrieval ví dụ kết hợp constrained semantic decoding; task của paper không phải LifeMate. [Synchromesh](https://arxiv.org/abs/2201.11227)

### D. Entity grounding + chuẩn hóa có nguồn: P0/P1

Tạo fact store **riêng từng sample** từ context, history hợp lệ và tool results. Mỗi fact gồm value/type, nguồn, thời điểm quan sát và trạng thái còn hiệu lực. Ví dụ tên phòng → device ID phải có bằng chứng của chính sample; x-ref chỉ ra loại tham chiếu, không tự cung cấp ID.

Resolver đối chiếu exact/alias rồi fuzzy candidates khi cần; ambiguous candidate phải lookup thêm nếu tool hỗ trợ hoặc ask đúng params. Không tự đặt ngưỡng fuzzy rồi ép mọi tên vào ID gần nhất. History phản ánh hành động đã áp dụng, không replay; trạng thái cũ có thể stale sau write.

Chuẩn hóa số tiền, thời gian theo `current_time` và timezone. Giữ nguyên raw text và provenance; số tiền, số tài khoản, ngày giờ chưa chắc chắn không được tự sửa bằng suy đoán. Invalidate/cache-refresh facts bị ảnh hưởng sau write, không mặc định đọc lại mọi tool vì thừa call làm giảm precision.

**Kỳ vọng:** cải thiện final state và argument exact match. Đo WRONG_ENTITY_ID, temporal/money errors, unnecessary reads, ask precision. Weak-regex speech hiện có chỉ phục vụ diagnostics, không đủ dùng làm resolver production.

### E. Constrained decoding: P1

Ràng buộc token khi generate để output theo một tool call đúng schema hoặc STOP. Tách ba lớp: syntax → schema → semantic grounding. Grammar đảm bảo cấu trúc trong phạm vi hỗ trợ; nó không đảm bảo đúng ý định, đủ tiền hay chọn đúng người.

XGrammar có hướng tích hợp Hugging Face bằng logits processor và structural tags cho tool protocols. Cần spike với **đúng tokenizer, native XML Qwen, EOS, quantization, package pins và GPU** hiện tại; không áp grammar JSON lên một format XML rồi sửa parser để che mismatch. Luôn giữ `env.validate()`.

Custom keywords `x-ref`, `x-match`, `x-one-of` không được mặc định có cùng nghĩa/hỗ trợ như chuẩn JSON Schema. Các semantic checks cần explicit implementation theo official catalog/runtime. Giữ đường STOP hợp lệ; không bắt model phải chọn tool khi task OOS.

**Kỳ vọng:** giảm structured_output_error và repair rounds. Đo parse/schema validity, score, compile/mask overhead và P95. [XGrammar HF quick start](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html), [tool calling/structural tags](https://xgrammar.mlc.ai/docs/latest/structural_tag/tool_calling_and_reasoning.html)

### F. Context engineering / task memory: P1/P2

Giữ current request, correction mới nhất, facts có provenance, unresolved slots và tool results cần thiết. Giữ các điều kiện phủ định như “đừng”, “trừ”, “thay bằng”. Khi render, phân biệt latest request, history đã thực thi và observation của lượt hiện tại.

Ưu tiên trích xuất có quy tắc và giữ raw trace để audit trước khi thêm LLM summarizer. Không âm thầm truncate history/schema; nếu compact phải có policy, coverage checks và ablation. Không dùng memory xuyên sample để lưu entity/state riêng tư hoặc câu trả lời của test.

**Kỳ vọng:** giảm prompt/prefill và history/reference errors. Đo theo history length, correction và multi-step slices. Nghiên cứu Lost in the Middle cho thấy vị trí thông tin ảnh hưởng khả năng sử dụng context ở các task khảo sát; lợi ích của cách compact này trên LifeMate vẫn là giả thuyết. [Lost in the Middle](https://arxiv.org/abs/2307.03172)

### G. Speech decoding không train lại: P1/P2

Giữ PhoWhisper baseline đã chọn; so greedy với beam nhỏ, contextual prompting bằng danh từ hợp lệ trong profile/history/catalog. Whisper có `prompt_ids`; phải thử đúng PhoWhisper checkpoint/package vì hỗ trợ API không bảo đảm tăng quality. Không lấy tên người/ID từ hidden environment để bias ASR. [Whisper generation API](https://huggingface.co/docs/transformers/model_doc/whisper)

Chỉ khảo sát N-best cho mẫu có entity bất định khi còn thời gian. Chọn transcript/action bằng acoustic evidence + schema/context, không chọn giả thuyết chỉ vì nó tạo được tool call hợp lệ. Lưu raw hypotheses và bản chọn; không sửa số tiền hoặc tên người thiếu bằng chứng.

Đo WER/CER cùng critical-entity error và **audio end-to-end score**. Beam/N-best tăng latency; WER tốt hơn chưa chắc state tốt hơn. ASR adapter hiện chọn theo WER/CER; có thể đánh giá downstream trên development sau đó nhưng cần giữ test holdout riêng để tránh selection bias.

### H. Hybrid symbolic planner / workflow templates: P2

Pattern như lookup → chọn ID → write có thể biểu diễn dependency graph/finite-state workflow; LLM chọn intent/slots, controller thực hiện các bước cần thiết. Template lưu quan hệ điều kiện/dependency, không lưu fixed ID hay bắt mọi task phải read trước write.

Ưu điểm kỳ vọng: ít LLM rounds, dừng rõ. Rủi ro: đổi schema, multi-app, correction và task ngoài template. Bắt đầu với 1–2 pattern phổ biến đã có evidence lỗi; fallback về agent khi điều kiện áp dụng không đạt. Không hard-code catalog/app list hoặc lời giải theo sample ID.

### I. Verifier, self-consistency, search: P3 / hạn chế

Ưu tiên verifier cục bộ deterministic: type/enum/required, installed tool, provenance ID, missing slots, completion ledger. Có thể chọn giữa vài **proposals chưa execute** nếu budget cho phép.

Không thử nhiều nhánh bằng nhiều ToolEnv của cùng sample, không rollback/discard calls để chọn nhánh thắng. Tree search dựa trên clone/reset simulator không phù hợp với bất biến một session/sample. Self-consistency hoặc thêm LLM critic tăng token/latency/model budget; chưa nên đưa vào baseline 30s.

Verifier không được dùng gold, target hash hoặc truy cập state để xác nhận task success trong inference. Không thêm read call sau mỗi write chỉ để “cho chắc”; chỉ đọc khi cần thiết theo semantics task.

### J. Runtime optimization: P0/P1 song song

Load model một lần/worker, greedy hoặc deterministic decoding, cache schema/grammar theo hash, precompute retrieval index, cache ASR history bằng audio hash + model/processor/decoding config. Cache không được chứa test labels hoặc state/ID của sample khác.

Đo lại prefill/decode/tool/ASR riêng và tổng audio latency. Quantization/kernel/backend phải so cả score và memory; fallback attention hiện tại giúp resource path nhưng không chứng minh runtime thi. Multi-GPU Kaggle giúp development; final runner phải kiểm chứng trên cấu hình một A30 theo rule đã xác minh.

Không thay model, decoding, retrieval và prompt đồng thời trong một ablation. Không tự thêm embedding/reranker/critic khi chưa tính tổng parameters, license/release date và memory.

## 4. Kiến trúc mục tiêu

```text
Audio + permitted context/history
  → PhoWhisper → transcript + raw acoustic evidence khi có
  → context builder + per-sample fact/task store
  → installed-tool filter
  → optional tool retrieval + dependency expansion
  → optional train-fit demonstration retrieval
  → small agent → one CALL | ASK | STOP
  → parser + schema validation + grounding guard
  → same ToolEnv.call() → actual observation
  → update/invalidate facts and task ledger → next step
  → env.get_tool_calls() + env.get_state_hash()
```

Runtime model không nhận gold/ref hashes hoặc environment files. Retrieval không gọi network ở final offline inference. STOP được quyết định từ input/observations/ledger, không từ gold-state comparison. ASK là tool thật theo catalog, không là câu hỏi free-text thay thế.

## 5. Experiment matrix và quyền sở hữu đề xuất

Giữ nguyên split IDs/seed, model/revision, ASR, scorer, budgets và hardware giữa hai nhánh; chỉ đổi yếu tố ghi trong hàng. Chạy text-oracle để tách lỗi agent rồi audio trên cùng IDs. Không dùng public leaderboard để chọn examples/thresholds. Nếu điều chỉnh lặp nhiều trên dev, cần một train-only holdout cuối chưa sử dụng; ghi rõ lại sample denominator/provenance.

| Thứ tự | Experiment | Owner / dependency | Definition of Done |
| --- | --- | --- | --- |
| E0 | Baseline hiện tại + trace đầy đủ | Data/Eval + Ops; scorer/split frozen | Full coverage IDs, failures giữ denominator, phân rã ASR/agent/time và lỗi chính |
| E1 | Harness contract parity notebook/Ops | Ops + LLM; E0 | Chung semantics STOP/ASK/deadline/failed call; model+official ToolEnv integration pass; không mất log |
| E2 | Grounding guard + fact store | LLM + Data; E1 | Trace nguồn ID; xử lý ambiguity/stale facts; giảm wrong-ID mà không tăng read/ask thừa |
| E3 | Constrained decoding | LLM + Ops; E1, compatibility spike | Native format/EOS pass; parse errors giảm; semantic score không giảm; overhead báo rõ |
| E4 | Lexical tool retrieval | Data + LLM; E0 | Recall@k, multi-app/dependency slices, tokens/latency/score so full catalog; fallback rõ |
| E5 | Retrieval 1–3 train-fit examples | Data + LLM; E4 | Index manifest chứng minh không dev/test, không copy IDs; paired score/latency so no-example |
| E6 | Context ledger/compaction | LLM; E2 | History/correction slices và retained constraints pass; memory/token reduction được đo |
| E7 | ASR prompting/beam nhỏ | Speech; E0 | Entity metric + downstream audio score, latency cùng hardware; không có gold ASR bias |
| E8 | Workflow 1–2 patterns | LLM + Ops; E0 error evidence | Applicability/fallback rõ; multi-step score/call count tốt hơn trên dev |
| E9 | Tích hợp các thay đổi có lợi | Research lead + Ops | Ablation từng phần và combined run; repeated determinism, offline Docker, submission coverage; resource/model/rule checks |

Không ấn định mức tăng score trước khi đo. Báo paired per-sample deltas và confidence interval khi đủ mẫu; lỗi critical write/OOS phải được kiểm tra riêng, không chỉ xem mean. Thời gian chạy 30s lấy từ nguồn luật đã xác minh; báo P50/P95/P99/max, timeout và hardware. Đo ASR+retrieval+prefill+decode+ToolEnv+export cho audio, startup ghi riêng và đối chiếu cách BTC tính.

## 6. Đề xuất scope trước hạn

1. Hoàn tất full baseline chạy được và giữ notebook training hiện tại làm control; chưa tăng quy mô distillation.
2. Làm E1/E2 và compatibility spike E3 trước. Nếu lỗi chủ yếu wrong intent thay vì format/ID, cập nhật ưu tiên theo trace.
3. E4 lexical retrieval khi prompt/catalog thực tế cho thấy cần; E5 chỉ sau khi kiểm soát split/leakage.
4. Chỉ làm E6–E8 khi error slices và thời gian chứng minh đáng đầu tư. Không đưa multi-agent runtime, graph database/GraphRAG, vector DB service hoặc tree search execution vào đường nộp lúc này.
5. Freeze giải pháp có evidence tốt; Docker offline và deterministic replay là deliverable riêng, không suy ra từ notebook train pass.

Đây là lựa chọn ưu tiên engineering dựa trên kiến trúc code và log team đã cung cấp, chưa phải kết luận hiệu quả thực nghiệm. Có thể xây control inference **không cập nhật weights** để đo riêng giá trị harness/RAG/grounding trước khi ghép với adapter đã train.

## 7. Nguồn và giới hạn survey

Nguồn kỹ thuật sơ cấp đã đọc: ReAct (2022/2023), ToolLLM (2023), Synchromesh (2022), Lost in the Middle (2023), XGrammar documentation và Hugging Face Whisper documentation qua các link tại từng mục. Các trang docs hiện hành cần pin phiên bản khi triển khai.

BFCL hỗ trợ cách nghĩ về evaluation tool use, relevance và multi-turn; không thay scorer state-hash của LifeMate và không dùng leaderboard BFCL để tuyên bố model thắng trên dữ liệu BTC. [BFCL — Berkeley](https://gorilla.cs.berkeley.edu/leaderboard.html)

Luật cuộc thi: [LifeMate portal](https://challenge.zalo.ai/portal/lifemate) chưa đọc được nội dung bằng web extraction. [Brief mirror](https://github.com/trtkiet/zalo-ai-challenge/blob/main/ZaloAI.md) là secondary; không dùng nó để xác nhận rule hiện hành. Survey không đọc initial state files, không tải weights, không sinh synthetic/test labels và không sửa code training/inference.
