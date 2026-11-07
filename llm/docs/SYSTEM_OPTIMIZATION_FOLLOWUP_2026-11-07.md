# LifeMate: survey bổ sung về cải thiện hệ thống

Ngày: 07/11/2026. Bổ sung cho `SYSTEM_METHODS_SURVEY_2026-11-07.md` và `speech/docs/ASR_IMPROVEMENT_SURVEY_2026-11-07.md`. Đây là nghiên cứu/thiết kế experiment; không sửa model, notebook hoặc chạy training/inference.

## Bối cảnh và quyết định

**TEAM OBSERVATION:** người dùng báo khoảng 40/100, Qwen3.5-4B đã fine-tune + PhoWhisper-medium, bottleneck nghiêng về ASR. Chưa có manifests/traces/gold-text oracle mới để định lượng từng nguồn lỗi. Không coi score này là official hoặc tự gán split/hardware.

**FACT/CODE:** source notebook decode PhoWhisper greedy; system scaffold nhận một transcript string qua `Transcriber`. Chưa có contract N-best/confidence, Vietnamese ITN tích hợp, prompt optimizer, workflow induction hoặc adaptive compute. Grounding hiện chỉ là explicit typed evidence bindings, chưa giải quyết việc ASR nghe sai tên.

**PROPOSAL:** trước khi tăng training, phân tách ba nhóm: audio nghe sai; nghe gần đúng nhưng chuyển sang slots/arguments sai; agent biết đủ thông tin nhưng thực hiện/stop sai. Nhóm khác nhau cần can thiệp khác nhau.

## 1. Vietnamese inverse text normalization (ITN)

Nguồn: [NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing), [release notes có Vietnamese ITN](https://github.com/NVIDIA/NeMo-text-processing/releases), [Vietnamese money grammar](https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/inverse_text_normalization/vi/taggers/money.py), [time grammar](https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/inverse_text_normalization/vi/taggers/time.py).

**RESEARCH/CODE:** release notes và source có Vietnamese TN/ITN, gồm money/time/date/telephone cùng các nhóm khác. Đây là grammar-based text processing, không tự sửa acoustic recognition hoặc giải quyết mọi ngữ cảnh tiếng Việt.

**Cách thử:** giữ raw ASR, thêm semantic candidates cho span chắc chắn, ví dụ “hai triệu đồng” → integer 2000000. Lưu raw span, canonical value và rule version. Tiền không rõ đơn vị, điện thoại có số 0 đầu, giờ thiếu AM/PM, tên “Mai” và từ “mai” cần policy riêng; không ép grammar output thành chân lý. Relative date vẫn cần current_time/timezone thực tế.

**Giá trị kỳ vọng:** giảm sai arguments khi ASR đã nhận đúng lời nói nhưng biểu diễn bằng chữ. Đo slot-value accuracy và downstream state, không dùng normalized WER để che việc execute sai tiền/ngày. Cần spike Pynini/OpenFst, packaging/offline và pin revision hợp lệ, chưa có compatibility certification trên Kaggle.

**Ưu tiên:** cao nếu money/date/phone là lỗi chính; thấp nếu audio mất hẳn tên hoặc động từ.

## 2. Uncertainty-aware spoken language understanding

Nguồn: [Improving Spoken Language Understanding By Exploiting ASR N-best Hypotheses](https://arxiv.org/abs/2001.05284), [Leveraging Large Language Models for Exploiting ASR Uncertainty](https://arxiv.org/abs/2309.04842).

**RESEARCH:** các công trình dùng nhiều giả thuyết ASR cho semantic understanding thay vì chỉ 1-best. Paper LLM nêu device-directed detection và keyword spotting, không phải benchmark LifeMate.

**Cách thử:** mở rộng ASR artifact thành raw hypotheses + acoustic scores + decoder provenance. Trong một model pass, đưa một tập nhỏ hypotheses và yêu cầu xác định intent/critical slots/điểm bất đồng trước khi đề xuất action. Có thể dùng internal `RequestIR` gồm action, target, amount/time, constraints, references, unresolved fields; IR không phải tool call BTC và không được execute trực tiếp.

Hai transcripts có từ khác nhau nhưng cùng action/slots không nhất thiết cần correction. Nếu khác số tiền/người nhận, cần thêm acoustic/grounding evidence hoặc đọc dữ liệu bằng tool hợp lệ khi thật sự cần. Tách language plausibility khỏi acoustic evidence; fluent text không chứng minh nghe đúng.

**Đo:** N-best oracle slot coverage trên train/dev, candidate semantic agreement, actual selected-slot accuracy, state match, extra LLM tokens. Oracle dùng gold chỉ là diagnostic, không có trong inference. Acoustic scores chưa calibrate không được xem là xác suất đúng.

**Ưu tiên:** cao nếu đúng intent/entity xuất hiện trong N-best; đây là thay đổi tại ASR→agent boundary, không chỉ thêm một text-correction model.

## 3. Complementary ASR + ROVER / disagreement analysis

Nguồn: [NIST ROVER paper](https://www.nist.gov/publications/post-processing-system-yield-reduced-word-error-rates-recognizer-output-voting-error), [SCTK implementation/docs](https://github.com/usnistgov/SCTK/blob/master/doc/rover/rover.htm).

**RESEARCH:** ROVER kết hợp outputs nhiều recognizers qua alignment và voting/rescoring. Lợi ích phụ thuộc error diversity; không có đảm bảo mọi ensemble tốt hơn từng model.

**Cách thử:** PhoWhisper-medium + một recognizer khác kiến trúc từ shortlist trước. Đầu tiên đo mẫu model A đúng/B sai, B đúng/A sai và cả hai sai; chưa vội chạy ensemble production. Nếu errors bổ sung nhau, thử chọn hypothesis hoặc fusion giữ critical spans. Chỉ hai recognizers dễ tie; không gọi đó là majority đáng tin nếu thiếu tie-break/calibration.

Không so trực tiếp avg_logprob khác kiến trúc như cùng thang đo. Không tự lấy tiền từ model này và người nhận từ model kia thành câu không có acoustic support. Có thể chạy ASR thứ hai có điều kiện sau khi calibrate trigger trên train-only data.

**Đo:** standalone vs combined slot accuracy/state/latency; oracle across systems chỉ là diagnostic tiềm năng, không phải điểm đạt được. Kiểm tổng parameters, checkpoint license/release date và memory của mọi model cùng hệ thống.

**Ưu tiên:** trung bình; chỉ triển khai nếu error complementarity đủ rõ và còn latency budget.

## 4. Offline prompt optimization bằng GEPA / DSPy

Nguồn: [GEPA paper](https://arxiv.org/abs/2507.19457), [GEPA repo](https://github.com/gepa-ai/gepa), [DSPy paper](https://arxiv.org/abs/2310.03714), [DSPy repo](https://github.com/stanfordnlp/dspy).

**RESEARCH:** GEPA cập nhật prompts từ feedback/trajectories qua reflective search; DSPy cung cấp cách lập trình và tối ưu LM pipelines. Không cần suy rằng phải đổi model hoặc dùng API ở final inference. Kết quả benchmark ngoài không dự báo mức tăng LifeMate.

**Cách thử:** đóng băng Qwen/ASR/decoder; chỉ tối ưu prompt cho hiểu ASR noise, chọn tool/arguments, xử lý history và stop. Feedback từ train-fit gồm local score có provenance + error taxonomy + actual logs. Dùng model phát triển hợp lệ để đề xuất prompt; final giữ prompt tĩnh, không chạy optimizer/teacher.

Tách `fit_search` và `fit_select` bên trong train-fit theo groups. Frozen validation dùng cho final audit, không đưa gold/traces của nó vào vòng phản tỉnh để học prompt. Không dùng public/private làm optimizer feedback hoặc confusion dictionary. Mỗi rollout riêng giữ toàn bộ calls; không reset/drop calls trong một rollout để chọn kết quả tốt.

Primary objective là score local được kiểm chứng; latency/call counts là diagnostics/constraints hoặc objective phụ phải ghi rõ, không tự gọi metric pha trộn là official. Giới hạn prompt length và search budget để tránh prompt bloat/overfit. GEPA/DSPy là hai lựa chọn triển khai, không cần chạy cả hai cho cùng một ablation.

**Ưu tiên:** cao nếu gold-text oracle cho thấy agent còn khoảng trống; không kỳ vọng prompt khôi phục thông tin âm thanh đã mất hoàn toàn.

## 5. Offline workflow memory thay cho chỉ retrieval ví dụ

Nguồn: [Agent Workflow Memory](https://arxiv.org/abs/2409.07429), [repo](https://github.com/zorazrw/agent-workflow-memory).

**RESEARCH:** AWM rút routine tái sử dụng từ trajectories và cung cấp workflow cho agent; thực nghiệm chính thuộc web navigation, không phải tool simulator LifeMate.

**Cách thử:** từ successful train-fit trajectories rút procedural patterns có preconditions/dependency/termination, delexicalize entity values. Khác example retrieval ở chỗ lưu quy trình khái quát và điều kiện áp dụng, không chỉ mẫu câu gần nghĩa. Dùng offline induction, freeze bank trước inference; không học thêm từ public/private runs.

Ví dụ trừu tượng: có entity ID đúng và còn hiệu lực → write trực tiếp; chưa có ID → lookup cần thiết → resolve ambiguity → write → dừng khi task hoàn tất. Không bắt mọi task read trước write, không thêm “verify read” vô điều kiện. Workflow chỉ là guidance, không phải lệnh thực thi cứng.

**Đo:** multi-step/history slices, extra/missing calls, premature STOP, token cost. So với no-memory và example-only retrieval cùng bank budget. Chỉ bật nếu agent errors là phần đáng kể.

## 6. Planning skeleton / dependency compiler

Nguồn: [ReWOO paper](https://arxiv.org/abs/2305.18323), [repo](https://github.com/billxbf/ReWOO), [LLMCompiler paper](https://arxiv.org/abs/2312.04511), [repo](https://github.com/SqueezeAILab/LLMCompiler).

**RESEARCH:** ReWOO tách planning khỏi observations để giảm prompt redundancy; LLMCompiler lập dependency plan và thực thi các function calls song song khi có thể. Các môi trường đó không chứng minh ToolEnv LifeMate thread-safe hoặc calls độc lập.

**Cách thử phù hợp:** planner tạo skeleton với unresolved variables; executor tuần tự bind biến từ actual tool results và validate từng bước. Khi observation khác dự kiến phải re-plan. Không suy diễn ID hoặc kết quả lookup từ plan.

Không đưa parallel execution vào cùng ToolEnv cho baseline này; ordering/log/hash phải giữ đúng. Không thêm fused tool không có trong catalog. Mục tiêu là giảm LLM rounds ở những pattern chắc chắn, không bypass feedback của write nhạy cảm.

**Ưu tiên:** sau khi baseline đáng tin; chỉ tối ưu nếu profiling cho thấy repeated LLM prefill/rounds là bottleneck.

## 7. Adaptive compute: tăng effort có chọn lọc

Nguồn: [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314).

**RESEARCH:** hiệu quả test-time compute phụ thuộc độ khó và model; nguồn khảo sát reasoning, không phải bằng chứng rằng “nghĩ lâu hơn” cải thiện tool execution tiếng Việt.

**Cách thử:** easy path một ASR + agent bình thường; hard path thêm một decode/hypothesis review trước execute khi có disagreement ở critical slots. Trigger dựa trên tín hiệu đã calibrate trên train-only data, không dựa model tự nói “tôi chắc chắn”. Có thể dùng cùng Qwen weights để tránh thêm model, nhưng mỗi round vẫn tốn latency.

Chỉ search giữa proposals chưa execute; không nhân bản/reset ToolEnv, không best-of-N execution rồi chọn hash. Reserve thời gian cho ToolEnv/export. STOP/ASK phụ thuộc semantics; hết budget không được coi là task success.

**Đo:** quality theo nhóm được/không được route, trigger false positives/negatives, conditional overhead, total P95/P99/max/timeout. Không tự nới giới hạn runtime hoặc dùng mean để che slow tail.

## 8. Clarification calibration: tham khảo có điều kiện

Nguồn: [KnowNo / Robots That Ask For Help](https://arxiv.org/abs/2307.01928).

**RESEARCH:** KnowNo dùng conformal prediction cho uncertainty/asking trong robot planning. Guarantee cần assumptions/calibration đúng setup; không tự chuyển sang ASR/tool simulator.

**Cách áp dụng giới hạn:** phân biệt thiếu thông tin thật với model chưa chắc, ASR nghe chưa rõ hoặc cần lookup. Re-decode/retrieve/grounding là lựa chọn khác với ASK. `sys.ask_missing_info` bị chấm như call bình thường, nên “confidence thấp → hỏi” có thể làm điểm giảm dù có vẻ thận trọng. Chỉ hỏi đúng tool/params theo schema và semantics. Không claim conformal guarantee khi chưa có calibration design phù hợp.

**Ưu tiên:** theo dõi ask precision/recall ngay; chưa đưa full KnowNo vào runtime trước hạn nếu không có evidence.

## Experiment sequence đề xuất

Giữ agent model, split IDs, scorer, decoder defaults và hardware cố định trong mỗi cặp so sánh. Dữ liệu ngoài phải hợp lệ về quyền sử dụng; không dùng public/private cho train/tune.

| Thứ tự | Experiment | Evidence để tiếp tục |
| --- | --- | --- |
| S0 | Text-oracle vs actual ASR + slot/error slices | Định lượng lỗi audio, normalization, agent thay vì suy từ WER chung |
| S1 | Raw ASR vs conservative ITN semantic candidates | Money/date/phone accuracy và actual state tốt hơn; không tăng critical wrong writes |
| S2 | 1-best vs uncertainty-aware semantic inference | Candidate coverage có ích, selected slots/score tăng, latency còn budget |
| S3 | Secondary ASR standalone → complementarity → selective ensemble | Có lỗi bổ sung nhau, không chỉ bảng WER ngoài; full model/system constraints pass |
| S4 | Fixed prompt vs one offline prompt optimizer | Fit-search/select tách biệt, validation audit và prompt provenance rõ |
| S5 | Example-only vs workflow memory | Giảm extra/missing calls, không ép workflow sai hoặc copy IDs |
| S6 | Sequential agent vs skeleton planner | Giảm model rounds/tokens nhưng dependency/state accuracy giữ hoặc tăng |
| S7 | Fixed effort vs calibrated adaptive compute | Tăng quality/latency trade-off trên frozen IDs, timeout không tăng ngoài budget |

Không cần chạy hết. Nếu ASR là bottleneck đã được xác minh, ưu tiên S1/S2 và decoding/contextual biasing/model control trong survey ASR trước. Nếu text-oracle cũng thấp, ưu tiên S4/S5 và harness/error analysis song song.

## Integration backlog, chưa triển khai

1. **Speech:** ASRResult contract giữ raw N-best/scores/model+decoder provenance; selective second-ASR adapter.
2. **Data/Eval:** critical-slot references/slices, train-only optimizer partitions, calibration subset, paired metrics và failure denominator.
3. **LLM:** candidate-aware prompt/IR, static optimized prompt, workflow bank và schema bindings thực.
4. **Ops:** deadline-aware router, private artifacts/versioning, parity notebook/runtime và offline checks.
5. **Research lead:** chọn experiment theo error evidence và tổng cost; không cộng dồn module khi chưa ablate.

Các đề xuất này chưa được benchmark trên LifeMate; không cam kết mức điểm tăng. Giữ nguyên nguyên tắc một environment mỗi sample/rollout, không truy cập initial state files, mọi executed call nằm trong output thật. Repo/doc main hiện hành cần pin version và kiểm model/data/license/release cutoff theo nguồn BTC trước triển khai.
