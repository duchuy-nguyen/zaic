# LifeMate: cải thiện ASR ngoài việc tiếp tục fine-tune

Ngày khảo sát: 07/11/2026. Không tải weights, chạy inference hoặc sửa training trong survey này.

## Evidence và phạm vi

- **TEAM:** người dùng báo khoảng 40/100 với Qwen3.5-4B đã fine-tune + PhoWhisper-medium, cho biết ASR là bottleneck. Chưa nhận được run manifest, frozen IDs, WER/CER mới hoặc gold-text oracle để xác minh mức tác động. Không dùng lại WER 34.16% của experiment cũ như WER hiện tại.
- **FACT/CODE:** `llm/notebooks/oneclick/worker.py::transcribe` hiện dùng `language='vi'`, `task='transcribe'`, `do_sample=False`, `num_beams=1`; chưa truyền contextual prompt hoặc xuất N-best trong đường này. Đây là source repo, không chứng minh session Kaggle của team có cùng config.
- **RESEARCH:** paper/model-card numbers dưới đây thuộc benchmark của tác giả, không phải score LifeMate. WER normalization khác nhau không được so trực tiếp.
- **PROPOSAL:** giữ model agent/prompt/controller/split cố định, thay từng yếu tố ASR rồi đo cả transcript và final state/tool F1.

## Shortlist có thể hành động

### 1. Contextual biasing: giữ PhoWhisper, sửa từ hiếm

[Can Contextual Biasing Remain Effective with Whisper and GPT-2? — Interspeech 2023](https://arxiv.org/abs/2306.01942), [WhisperBiasing code](https://github.com/BriansIDP/WhisperBiasing).

Paper dùng TCPGen và training component biasing, giữ weights Whisper cố định. Repo có training/decoding scripts; không phải flag hotwords bật là có TCPGen. Benchmark English của repo không chứng minh tương thích PhoWhisper/Vietnamese tokenizer.

**Đề xuất:** trước khi port TCPGen, thử contextual prompting bằng danh sách ngắn từ profile/history/catalog được phép. Đây là baseline đơn giản riêng, không gọi là tái lập TCPGen. Bias list không được lấy từ gold hoặc initial state; quá nhiều tên gây over-bias.

### 2. Entity RAG: truy xuất tên hợp lệ để sửa lỗi nghe

[Retrieval Augmented Correction of Named Entity Speech Recognition Errors — 2024](https://arxiv.org/abs/2409.06062).

Paper truy xuất entity rồi dùng LLM đã adapt để sửa ASR; báo 33–39% relative WER reduction trên synthetic rare-music-entity test sets, không phải tiếng Việt/LifeMate. Chưa xác minh một implementation công khai của chính paper trong lần khảo sát này.

**Đề xuất:** entity index riêng sample từ context/history hoặc tool results đã được phép. Lexical/character/phonetic candidate retrieval là baseline cần đo, không tương đương vector retriever của paper. Không ép mọi span sang entity gần nhất; tiền/số tài khoản không được tự sửa vì câu nghe hợp lý hơn. Lookup phát sinh vẫn là call bị chấm, không gọi tràn lan để xây index.

### 3. N-best/lattice-constrained correction

[ASR Error Correction using Large Language Models — 2024, revised 2025](https://arxiv.org/abs/2409.09554).

Paper nghiên cứu N-best inputs và constrained decoding để giới hạn correction; không chỉ cho LLM viết lại transcript tự do. Chưa xác minh repo chính thức của paper này; không gắn repo correction khác làm implementation của nó.

**Đề xuất:** xuất 3–5 hypotheses cùng acoustic scores, đo oracle WER/entity accuracy trước khi viết reranker. Oracle chọn bằng gold chỉ là diagnostic offline trên train/dev, tuyệt đối không phải inference. So reranking với 1-best; score khác decoder không mặc định cùng thang. Beam size >1 không tự có N-best output: cần lấy nhiều sequences đúng API. Không execute nhiều ToolEnv branches để chọn transcript.

### 4. ChunkFormer-RNNT: đối chứng model khác kiến trúc

[ChunkFormer paper, ICASSP 2025](https://arxiv.org/abs/2502.14673), [repo](https://github.com/khanld/chunkformer), [RNNT model card](https://huggingface.co/khanhld/chunkformer-rnnt-large-vie).

RNNT card ghi 113M params, CC-BY-4.0; báo normalized WER VIVOS 2.49 và Common Voice 5.18. Đây là author-reported benchmark, không phải kết quả trên lệnh thoại/code-switch của team. [CTC card](https://huggingface.co/khanhld/chunkformer-ctc-large-vie) là checkpoint khác: 110M, **CC-BY-NC-4.0**. Không suy license weights từ license repo.

**Đề xuất:** benchmark RNNT pretrained, chưa train, như đối chứng ưu tiên. Repo nêu RNNT release 10/2025; vẫn cần pin exact checkpoint revision trước cutoff BTC và kiểm license/dependencies. Không mặc định `transformers.pipeline` hỗ trợ checkpoint này; dùng inference code được tác giả cung cấp. Long-form efficiency của paper không chứng minh speed/quality trên short commands.

### 5. Gipformer 1.5: đối chứng nhỏ thứ hai nếu còn thời gian

[Repo](https://github.com/ggroup-ai-lab/gipformer), [model card](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt).

Model card mô tả Zipformer/RNNT với ONNX và MIT license; repo có test domains/dialects, một phần là private call-center sets. Các số đó là author-reported. Cần kiểm checkpoint ngày phát hành, parameter count thật và code-switch trên dev LifeMate trước khi chọn; không suy “68M” trong tên là số params đã audit.

**Đề xuất:** thêm một nhánh inference-only sau ChunkFormer nếu runtime dễ tích hợp. Không đồng nhất Gipformer, VietASR của paper 68M và SDK VietASR hiện hành; tên gần nhau không chứng minh cùng model. Không chọn một bảng benchmark có normalization khác làm quyết định thắng.

### 6. faster-whisper: hạ chi phí decoding để thử beam/hints

[Repo và conversion guide](https://github.com/SYSTRAN/faster-whisper), [source hotwords/prompt options](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py).

Repo hỗ trợ convert Transformers-compatible Whisper checkpoints; source có initial_prompt/hotwords. Đây là backend inference, không tự tăng quality. Giữ cùng beam/options khi so tốc độ. Với LoRA cần merge/export full checkpoint hợp lệ rồi kiểm conversion parity; không đưa adapter directory vào converter và mặc định đó là tuned model. Hotwords không có hiệu lực khi dùng prefix theo source hiện tại.

**Đề xuất:** thử HF beam/prompt trước; chỉ convert khi profiling cho thấy decoding time đáng tối ưu. Kiểm raw/tuned checkpoint, tokenizer, generation options, quantization và output parity trên fixed IDs.

## PhoWhisper-large có nên thử?

Có thể làm đối chứng, nhưng chưa ưu tiên thay medium ngay. [VinAI model table](https://github.com/VinAIResearch/PhoWhisper) ghi medium 769M, large 1.55B; VIVOS WER 4.97→4.67 và VLSP Task-1 14.12→13.75. Các cải thiện này thuộc benchmark gốc, không dự báo mức tăng trên LifeMate; tài nguyên tăng đáng kể. Kiểm total system params theo rule BTC trước khi chốt inference models.

## Thiết kế đo trước khi chọn giải pháp

1. **A0:** cùng agent đã fine-tune, gold transcript train-only dev → local evaluator. **A1:** cùng agent, current ASR transcript → cùng evaluator/IDs. Khoảng cách A0–A1 đo tác động transcript corruption trong setup đó, không chứng minh mọi lỗi trong khoảng cách đều sửa được bằng ASR mới.
2. So PhoWhisper base vs tuned checkpoint cùng decoding. Xem adapter có cải thiện entity/code-switch hay chỉ làm WER chung tốt hơn; đừng train tiếp khi chưa biết.
3. Kiểm audio path/rate/channels/duration, không truncate; task transcribe thay vì translate, không rơi mất history audio. Kiểm sample mapping và normalization trước khi kết luận model nghe kém.
4. **A2:** current PhoWhisper beam 1 vs 3/5; **A3:** contextual prompt on/off; đổi riêng từng yếu tố. `language='vi'` trên code-switch là một yếu tố cần ablation, không tự tắt hay ép English toàn bộ.
5. **A4:** RNNT pretrained, cùng audio/IDs/normalization, không tuning theo public/private. So critical-entity error và agent score, không chỉ WER.
6. **A5:** N-best diagnostic → reranker hoặc entity RAG khi errors cho thấy phù hợp. Nếu candidates đều thiếu đúng entity, không kỳ vọng selection-only khôi phục được.
7. **A6:** chỉ port/train TCPGen nếu từ hiếm là lỗi chính và còn thời gian; tiếng Việt tokenizer/bias-list, added params, latency đều cần đo.

Metrics: raw WER/CER + normalized WER/CER có version; name/address/money/date/phone/code-switch slices; deletion/substitution/insertion; gold-state match; local gated tool F1; audio total P50/P95/P99/max; timeout. Normalization phục vụ metric không được rewrite giá trị execute. Confidence/avg_logprob chỉ là tín hiệu cần calibrate trên train-only dev, không là xác suất đúng đã được chứng minh.

Nếu train correction hoặc agent chống nhiễu: cặp ASR hypothesis → transcript/tool target chỉ lấy từ train-fit. Ưu tiên hypotheses out-of-fold hoặc held-out khỏi ASR adaptation để tránh train corrections trên lỗi quá dễ. Dev chỉ evaluate/chọn cấu hình có kiểm soát; không dùng public/private để label, train hoặc xây confusion dictionary.

## Ghép vào khung hiện có

Scaffold `Transcriber.transcribe(...)` hiện trả một string. Beam/prompt hoặc thay ASR có thể giữ interface. N-best/ASR scores cần contract mới giữ raw hypotheses, selected hypothesis và provenance; chưa có trong scaffold, chưa triển khai ở survey này. Entity source phải đến từ đúng sample, không đọc environment files. Second-pass correction bằng Qwen hiện có vẫn thêm model round/latency; không mặc định free vì dùng lại weights.

Quyết định tạm thời: **giữ medium làm control; thử decoding/context → benchmark RNNT → entity-aware correction; chưa mở thêm training lớn**. Chưa có evidence cam kết tăng từ 40 lên một mức điểm cụ thể. Các source đọc từ main/current docs phải được pin release/revision và đối chiếu rules/allowed model dates trước triển khai.
