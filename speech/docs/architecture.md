# Báo cáo kiến trúc và tiến độ phần Speech — LifeMate

Ngày cập nhật: 08/11/2026. Lượt full TRAIN hoàn tất ngày 07/11/2026.

Bản tích hợp repo: code ở `speech`; đặt `LIFEMATE_SPEECH_WORK_DIR` để dùng work directory riêng. Báo cáo chứa thống kê tổng hợp; dữ liệu/run artifacts chi tiết lưu private, không nằm trong commit. Xem [hướng dẫn chạy](../README.md) và [báo cáo full TRAIN](FULL_TRAIN_ASR_REPORT_2026-11-07.md).

## 1. Vị trí của Speech trong toàn hệ thống

Theo kiến trúc dự kiến trong repo, có thể chia toàn hệ thống LifeMate thành **6 khối chức năng chính**. Đây là cách nhóm chức năng để trình bày; không có nghĩa toàn bộ hệ thống đã triển khai xong.

Ở mức khái quát có thể gom thành 3 khối Speech → LLM → ToolEnv. Sơ đồ 6 khối dưới đây tách thêm phần nạp dữ liệu, dựng prompt, điều phối và ghi đầu ra để làm rõ trách nhiệm; hai cách đếm khác mức chi tiết, không phải hai kiến trúc mâu thuẫn.

| Khối | Nhiệm vụ |
|---|---|
| **1. Input/Data Loader** | Đọc audio yêu cầu hiện tại, history, context và danh sách tool được phép dùng. |
| **2. Speech** | Chuyển audio hiện tại và audio user trong history thành văn bản tiếng Việt. |
| **3. Prompt Builder** | Ghép transcript với context, lịch sử assistant và schema tool thành đầu vào cho LLM. |
| **4. LLM — Qwen3-4B** | Hiểu yêu cầu, đề xuất gọi tool hoặc hỏi thông tin còn thiếu. |
| **5. Runtime + ToolEnv** | Kiểm tra và thực thi tool, trả kết quả về LLM để tiếp tục xử lý. |
| **6. Output Writer** | Ghi các tool call đã thực thi và state hash thành kết quả nộp bài. |

Luồng xử lý tổng thể:

```text
Input/Data Loader
    ├── Audio current + user history → Speech → Transcript ──┐
    └── Context + assistant history + schema tool ───────────┤
                                                            ↓
                                                      Prompt Builder
                                                            ↓
                                                           LLM
                                                            ↕
                                                     Runtime + ToolEnv
                                                            ↓
                                                       Output Writer
```

Speech nằm ở đầu luồng xử lý, cung cấp văn bản cho Prompt Builder. Context và schema tool đi trực tiếp từ Input sang Prompt Builder.

## 2. Kiến trúc bên trong khối Speech

Khối Speech gồm **4 phần chính**:

| Phần | Xử lý gì | Code chính |
|---|---|---|
| **Đọc và kiểm tra audio** | Lấy đường dẫn từ JSONL; kiểm tra định dạng, độ dài và lỗi file. Xử lý cả current và toàn bộ user history. | `speech/src/lifemate_speech/io_utils.py` |
| **Cache** | Dùng SHA256 nội dung audio, revision/config ASR và normalization version để tìm kết quả đã phiên âm. | `speech/src/lifemate_speech/cache.py` |
| **ASR** | Khi cache chưa có, chạy PhoWhisper-small theo batch được sắp xếp theo duration. | `speech/src/lifemate_speech/asr.py` |
| **Normalization và xuất transcript** | Chuẩn hóa NFC và khoảng trắng; xuất văn bản cùng metadata, timing, thông tin cache và lỗi. | `speech/src/lifemate_speech/normalize.py`, `speech/scripts/transcribe.py` |

Luồng xử lý Speech:

```text
Audio references từ JSONL
          ↓
Kiểm tra audio + tính SHA256
          ↓
     Tra cứu cache
      ├── Hit  → Lấy transcript đã lưu ──┐
      └── Miss → PhoWhisper-small ───────┤
                                        ↓
                           Normalize NFC + whitespace
                                        ↓
                          Transcript + metadata + lỗi
```

Cache chủ yếu hỗ trợ phát triển: chạy lại, resume và dùng cùng transcript artifacts để so chất lượng B1/B3-A. Khi gặp audio ẩn, không giả định sẽ có cache hit. Sơ đồ trên mô tả pipeline có hỗ trợ cache, không khẳng định cache giúp đạt latency khi thi. Theo plan, phép đo runtime cần **live uncached end-to-end timing**, gồm cả current và user history, trên hardware mục tiêu.

### Cấu hình và nguyên tắc

- Model: `vinai/PhoWhisper-small`.
- Revision: `a86b604c346caf7148c37512eafe783a16420adb`.
- Giải mã greedy: `num_beams=1`, `do_sample=False`, `language="vi"`, `task="transcribe"`, seed cố định.
- Không fine-tune ASR, không sửa kết quả theo transcript chuẩn.
- Normalization chỉ NFC và khoảng trắng; giữ nguyên số, dấu, chữ hoa/thường và dấu câu.
- Audio thiếu/hỏng vẫn được ghi nhận trong mẫu số và báo lỗi rõ ràng.
- Audio trên 30 giây hiện bị từ chối bằng lỗi `AudioTooLong`, không âm thầm cắt cụt; chưa triển khai chunking.
- Cache hit được tách khỏi inference mới khi báo cáo timing.

## 3. Phạm vi phụ trách và đầu ra bàn giao

Phần Speech chịu trách nhiệm chuyển giọng nói thành văn bản và đo chất lượng nhận dạng. Speech không quyết định gọi tool, không sửa tham số nghiệp vụ và không quản lý trạng thái app.

Đầu ra bàn giao:

- Transcript của audio current và từng lượt user trong history.
- Danh tính ghép dữ liệu: `sample_id`, `kind`, `turn_index`, `audio_path`.
- `text_raw`, `text_norm`, audio SHA256, model/revision và normalization version.
- Thông tin cache, timing theo batch và lỗi nếu có.
- Các artifacts: `transcripts.jsonl`, `errors.jsonl`, `run_manifest.json`.

`turn_index` là chỉ số zero-based trong toàn bộ history, có tính cả vị trí của lượt assistant; không phải số thứ tự riêng của lượt user.

## 4. Audit và evaluation ngoài luồng inference

Các công cụ này phục vụ kiểm tra dữ liệu và đánh giá trong giai đoạn phát triển:

| Công cụ | Nhiệm vụ |
|---|---|
| `audit_audio.py` | Kiểm tra inventory, định dạng, duration, silence, clipping và file lỗi. |
| `audit_supplement.py` | Thống kê percentile, RMS, số audio/sample và trùng bytes SHA256. |
| `metrics.py` | Tính WER/CER micro và macro, giữ mẫu số đầy đủ. |
| `entities.py` | Tạo weak labels và diagnostics cho tiền, giờ, ngày, điện thoại, tên/liên hệ, địa chỉ, từ tiếng Anh và thời gian tương đối. |
| `eval_asr.py` | Ghép reference–hypothesis theo `(sample_id, kind, turn_index)` và xuất báo cáo TRAIN. |

Reference `audio_text` chỉ dùng trong evaluation TRAIN và review riêng sau inference. Không đưa gold vào ASR/cache hoặc dùng làm fallback.

Quy tắc hiện tại: `ENTITY_RULES_VERSION=weak-regex-v3`. So khớp tiền nới lỏng và clock-value được báo cáo riêng, không sửa transcript, normalization hoặc WER/CER. Entity regex là weak labels, không phải nhãn thực thể chuẩn đã được con người xác nhận.

## 5. Tiến độ và kết quả thực tế

| Hạng mục | Trạng thái |
|---|---|
| Full waveform audit | Đã kiểm tra 6.969 WAV của train và public; không file thiếu/hỏng hoặc bị gắn cờ theo ngưỡng audit hiện tại. |
| ASR provider, normalization, cache, batching, resume | Đã triển khai và smoke test. |
| Smoke current | 5 mẫu train, 5 audio; cache lạnh/nóng thành công. |
| Smoke có history | 4 mẫu train, 20 audio gồm 4 current + 16 history; có mẫu 10 lượt user history. |
| Evaluation smoke | 25 audio thuộc 9 mẫu train, không thiếu hypothesis hoặc có lỗi ASR. |
| Quy tắc đánh giá v3 | Đã sửa tiền có phần lẻ/rưỡi, giờ kém/rưỡi, ngày dạng chữ và xử lý Mai kèm uncertainty. |
| Tests | 19 fixture test đã qua trong môi trường CUDA trước lượt full. |
| Full TRAIN ASR/evaluation | **Đã hoàn tất:** 5.036/5.036 audio, 1.545 current + 3.491 history; 0 lỗi xử lý, không thiếu/rỗng hypothesis. |
| Phiên âm PUBLIC để bàn giao LLM | **Chưa chạy** 1.933 audio (634 current + 1.299 history). |
| CUDA/RTX 4060, GPU smoke và peak VRAM | **Đã kiểm chứng:** CUDA 12.4 / PyTorch 2.6.0+cu124, GPU FP16 batch 2; full mất 48,9 phút, peak allocated/reserved 744/796 MiB. |

### Chất lượng trên toàn TRAIN — kết quả hiện tại

| Nhóm | Số audio | WER micro | CER micro |
|---|---:|---:|---:|
| Tất cả | 5.036 | 46,76% | 23,23% |
| Current | 1.545 | 45,08% | 22,63% |
| History | 3.491 | 48,31% | 23,81% |

Đây là inference/evaluation với trọng số nguyên bản, không phải huấn luyện. WER/CER giữ nguyên case và dấu câu. Transcript, audio SHA256, coverage và source/config đã kiểm chứng; 22 audio trùng smoke GPU có cùng transcript. Xem [báo cáo tổng hợp](FULL_TRAIN_ASR_REPORT_2026-11-07.md) để biết macro metrics, lỗi thực thể kèm mẫu số và provenance.

Smoke 20 mẫu/22 audio CPU FP32 so với GPU FP16 có 1 audio khác nhau. Mức lệch đã được chấp nhận trước full. Hai lượt dùng cùng source/weights, chỉ khác device/dtype; không coi đây là ablation chỉ thay device.

### Chất lượng trên bộ smoke 25 audio — số liệu lịch sử

| Nhóm | Số audio | WER micro | CER micro |
|---|---:|---:|---:|
| Tất cả | 25 | 35,36% | 16,30% |
| Current | 9 | 34,64% | 16,41% |
| History | 16 | 36,14% | 16,18% |

WER giữ nguyên case và dấu câu, tách token theo whitespace; CER tính cả khoảng trắng. Đây là kết quả trên bộ smoke nhỏ, chưa đại diện toàn TRAIN. Timing CPU hoặc cache không dùng để suy ra latency thi đấu trên GPU.

## 6. Ràng buộc tuân thủ

- DATA_ROOT giữ nguyên, chỉ đọc; output đặt trong `speech_work`, code ở `speech`.
- Không dùng `audio_text`, label hoặc final_state làm fallback/hint cho ASR. Gold transcript chỉ dùng trong evaluation TRAIN và review riêng.
- Không sửa transcript ASR theo gold; không đổi số, giờ, tên hoặc bỏ dấu trong normalization. Một hàm normalize dùng chung cho inference/evaluation.
- PhoWhisper-small giữ nguyên revision, không fine-tune, không thay config giải mã đã chốt trong bước chuẩn bị full nếu chưa thông báo và xác nhận.
- Không tạo nhãn, huấn luyện, chọn ngưỡng hoặc chỉnh quy tắc từ public_test. Có thể phiên âm public bằng cấu hình đã chốt để bàn giao; không tính WER/CER public vì không có gold.
- Giữ đầy đủ current và mọi user history trong mẫu số; file thiếu/hỏng hoặc inference lỗi phải có record rõ ràng.
- Regex/relaxed matching chỉ dùng đánh giá, không sửa transcript hoặc truyền kết quả phân tích gold vào luồng inference.
- Chỉ thực hiện ASR optimization ở giai đoạn được LEAD duyệt theo evidence; hiện chưa làm B4.

## 7. Điểm cần theo dõi

### 7.1. Độ trễ theo số audio mỗi sample

Trong smoke CPU, `train_1594` có **11 audio** (1 current + 10 user history), tổng thời gian ASR **phân bổ từ batch khoảng 43,855 giây**. Có 1/9 sample smoke có tổng phân bổ trên 30 giây. Nguồn: metrics smoke v3 - artifact local: `speech_work/eval_smoke_v3/metrics.json`, mục `latency.sample_totals`.

Đây không phải wall time end-to-end của sample và không chứng minh GPU sẽ vượt mốc 30 giây trong plan. Timing này chưa gồm model load, hash/header I/O và các khối LLM/tool/output. Cần đo live không cache trên hardware mục tiêu cùng OPS.

Chi phí không chỉ phụ thuộc số file: Whisper short-form đệm tới 30 giây khiến số audio và cách chia batch ảnh hưởng lớn; duration vẫn ảnh hưởng đọc audio/feature extraction, còn độ dài output, device và backend cũng ảnh hưởng thời gian. Không kết luận “độ trễ không phụ thuộc độ dài audio”.

Full audit cho thấy 198 conversation trong train + public có tổng độ dài tín hiệu current/history trên 30 giây. Độ dài tín hiệu cũng không phải latency inference; đây là nhóm cần chú ý khi thiết kế phép đo runtime.

Trên full TRAIN GPU, latency phân bổ mỗi file có p50/p95/max khoảng 0,559/0,975/1,576 giây; tổng phân bổ ASR mỗi sample khoảng 1,499/5,360/8,499 giây. Không sample nào có tổng phân bổ ASR trên 30 giây, nhưng đây vẫn không phải wall time end-to-end hoặc chứng nhận giới hạn thi đấu. Chi tiết ở [báo cáo full](FULL_TRAIN_ASR_REPORT_2026-11-07.md).

### 7.2. Từ “mai” và thời gian tương đối

Trên bộ smoke 25 audio, weak diagnostic ghi nhận **3/4 câu có phrase tham chiếu “mai” bị thiếu/sai phrase**; không phải thống kê 3/4 token hoặc tỷ lệ lỗi toàn train. Các cụm khác: “chiều nay” sai 1/2 câu, “hôm nay” sai 1/1 câu. Xem `relative_time.per_reference_phrase` trong metrics - artifact local: `speech_work/eval_smoke_v3/metrics.json`.

Nhầm “mai” thành “may” hoặc mất từ thời gian có thể làm sai yêu cầu nghiệp vụ. Tuy nhiên, mẫu nhỏ và “Mai” có thể là tên riêng. V3 loại trường hợp sau danh xưng cô/chị/bác/anh…, còn bare Mai vẫn gắn uncertainty. Cần xác nhận trên TRAIN lớn hơn và nghe lại trước khi dùng evidence cho quyết định B4; không tự sửa transcript hiện tại.

Full TRAIN đã bổ sung evidence: phrase “mai” thiếu/sai ở **144/347 câu**; toàn bộ relative-time diagnostic sai **479/1.358 câu**. Đây vẫn là weak labels, không phải annotation của con người hoặc bằng chứng mọi lỗi nghiệp vụ đều do ASR. Cần nghe/review và paired B0–B1 trước khi quyết định B4.

### 7.3. Giới hạn của weak parser

Phần lẻ tiền thiếu đơn vị, rưỡi sau nhiều đơn vị, giờ thiếu sáng/chiều và ngày thiếu tháng/năm vẫn có trường hợp chưa chắc. Guard tiền theo yêu cầu B nghiêm ngặt có thể reject cách viết VND nguyên tương đương khi chỉ một phía có fraction marker. Đây là giới hạn của diagnostic, không phải bằng chứng ASR chắc chắn sai. Danh sách cụ thể ở parser_uncertainties.md - artifact local: `speech_work/eval_smoke_v3/parser_uncertainties.md`.

## 8. Bước tiếp theo đang chờ xác nhận

1. Bàn giao bộ artifact TRAIN và provenance qua storage private; dùng báo cáo aggregate trong repo để review tiến độ.
2. Review/nghe các worst samples và parser uncertainties, tập trung tên/địa chỉ/tiền/“mai”; giữ nguyên transcript gốc.
3. Nối provider Speech với contract adapter OPS, chốt frozen DATA IDs và bảo đảm current/history, lỗi, deadline không bị rơi.
4. Sau khi B0 được freeze, phối hợp LLM/OPS/DATA chạy B1 trên frozen validation, lấy S0_AUDIO và so paired với S0_TEXT.
5. Khi được xác nhận phạm vi, chạy ASR PUBLIC đủ 1.933 audio bằng cấu hình đã chốt để bàn giao current/history transcripts và provenance. Không dùng public để tune hoặc tính WER/CER; việc này chưa thực hiện.
6. Phối hợp OPS đo end-to-end không cache trên hardware mục tiêu; phép đo full TRAIN ASR theo global batches không thay thế bước này.

Đã hoàn tất GPU smoke, full TRAIN inference và full TRAIN evaluation. Bàn giao tích hợp B1, public inference và target-hardware end-to-end vẫn là các bước tiếp theo; lượt full này không tự hoàn thành các gate đó.

## 9. Tài liệu và kết quả liên quan

- [Project plan](../../LifeMate/PROJECT_PLAN_2026-11-03.md).
- [Báo cáo tổng hợp full TRAIN GPU](FULL_TRAIN_ASR_REPORT_2026-11-07.md).
- [Lock môi trường Windows CUDA đã kiểm chứng](../requirements-cuda-win-py312.lock.txt).
- Báo cáo full audit - artifact local: `speech_work/audit/audio_audit_report.md`.
- Báo cáo smoke và evaluation v2 - artifact local: `speech_work/step4_final_smoke_report.md`.
- Báo cáo sửa quy tắc v3 - artifact local: `speech_work/eval_smoke_v3/v3_review_report.md`.
- 10 ví dụ mỗi loại thực thể - artifact local: `speech_work/eval_smoke_v3/entity_examples.md`.
- Các trường hợp parser chưa chắc - artifact local: `speech_work/eval_smoke_v3/parser_uncertainties.md`.
- Metrics smoke v3 - artifact local: `speech_work/eval_smoke_v3/metrics.json`.
- Kiểm chứng ASR/normalization/transcript giữ nguyên - artifact local: `speech_work/eval_smoke_v3/evaluation_only_validation.json`.
