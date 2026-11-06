# Chạy LifeMate bằng một notebook trên Kaggle

**Bản v6 — kiểm tra sớm và tiếp tục từ kết quả v5:**

```python
RUN_NAME = 'lifemate_pilot_v6'
RESUME_FROM = ''
REUSE_ASR_FROM = '/kaggle/working/lifemate/lifemate_pilot_v5'
REUSE_TEACHER_FROM = '/kaggle/working/lifemate/lifemate_pilot_v5'
```

Dùng hai đường dẫn reuse khi run v5 còn đủ completed stages/artifacts; nếu chạy từ đầu thì để trống. Teacher reuse giữ đúng model/revision nguồn, re-score với cùng split/reference; không sinh teacher lại, không đổi tên nguồn 9B thành 3.8. Notebook tự chặn nếu dữ liệu/config không tương thích. Trước khi thay notebook/session, lưu private archive của v5 bằng `save_resume_archive()` trong notebook đang mở.

Đầu pipeline v6: `preflight → stack_check → prepare → reuse_asr → references → reuse_teacher → student_precheck`. Stack check dùng tiny random model để kiểm tra NF4/Trainer/AMP/save/resume/reload thật; precheck dùng full student tại context cấu hình và gold-fit length coverage. Hai kiểm tra GPU này chạy trước ASR/teacher jobs dài. Các stages hoàn tất từ source chỉ được skip sau khi xác minh và ghi provenance.

Student dùng exact attention theo khối query 128 và loss theo 32 target tokens để giới hạn temporary tensors; backward recompute. Giữ đủ context, không thay bằng sliding window hay cắt labels. Frozen embedding/head khôi phục FP16; norms/adapters FP32. Hai smoke steps phải có gradient hữu hạn và optimizer updates. Xem [audit và giới hạn kiểm chứng](AUDIT_TRAINING_STACK_2026-11-06.md).

File cần import: **`LifeMate_OneClick.ipynb`** trong cùng thư mục. Notebook đã nhúng code cần thiết; không cần clone repository hay upload các file `.py`.

## Notebook train những gì?

| Thành phần | Cách dùng |
| --- | --- |
| PhoWhisper-medium | LoRA từ WAV + transcript thật của BTC train; so với bản gốc trên validation, chỉ chọn adapter khi WER/CER cải thiện |
| Qwen3.5-9B | Teacher đóng băng, quantize NF4 lúc load; chỉ inference để sinh trajectory trên fit subset. Thay teacher 27B sau khi team gặp OOM và CPU spill trên T4 x2 |
| Qwen3.5-4B | Student QLoRA một lượt trên BTC gold + next-action examples từ teacher đã qua lọc |

Đây là **sequence-level distillation**: học output/tool trajectory của teacher, không dùng KL giữa teacher/student logits. Teacher nhận transcript, context và tool results; không nhận gold labels hay initial app state. Teacher rollout chỉ được giữ nếu đạt state-hash và local tool-call score gate. Exact duplicate examples được dedup; xem `examples_report.json` để biết teacher thực sự bổ sung bao nhiêu examples mới. Không mặc định distillation sẽ tốt hơn gold-only SFT.

Chỉ dùng **BTC train** cho fit/validation. Split được cố định bằng seed và group theo audio trùng, hội thoại/utterance trùng và near-duplicate; thiếu speaker IDs nên chưa bảo đảm speaker-disjoint. Không dùng public/private test để train, sinh nhãn hoặc tuning. Không dùng tập synthetic cũ của repository trong notebook này.

## 1. Setup Kaggle

1. Tạo notebook **Private**, import `LifeMate_OneClick.ipynb`.
2. Chọn accelerator **GPU T4 x2**, bật **Internet** để cài dependencies và tải model. Hai T4 có 16 GB mỗi GPU; không phải một GPU 32 GB. Teacher được shard hai GPU; ASR/student training dùng một GPU.
3. Attach data kit BTC dưới dạng **Private**: ZIP `train+public_test.zip` nguyên bản hoặc folder đã giải nén. Cần đầy đủ `train.jsonl`, catalog, audio, `zaic_tools` và environment để ToolEnv hoạt động. Notebook không đọc trực tiếp initial state files. Không upload dữ liệu lên dịch vụ nếu điều khoản BTC không cho phép.
4. Dùng Kaggle image Python 3.11–3.13 có NVIDIA GPU/driver. Notebook tạo venv **cô lập** `/tmp/lifemate_training_env_v2`, cài `torch==2.9.1`, `torchvision==0.24.1`, `torchaudio==2.9.1` từ CUDA 12.6 index chính thức rồi cài các dependencies đã pin. Không cần kernel có sẵn torch, không thay packages của kernel. Venv v1 không được dùng lại. Bootstrap loại bỏ `PYTHONPATH`/`PYTHONHOME` kế thừa; worker dùng `python -I` và tự thêm đường dẫn bundle, tránh hook `sitecustomize` được inject qua PYTHONPATH. Stack bổ sung `wrapt==1.17.3`. Kiểm tra import và CUDA allocation thật trước khi tải model. Đây là thay đổi sau lỗi thiếu `wrapt`/`torch` trên image Python 3.13.
5. Model public không bắt buộc HF token. Nếu cần token, thêm Kaggle Secret rồi đặt `HF_SECRET_NAME` bằng tên secret; không ghi token trực tiếp vào notebook.

Nếu image báo lỗi `ensurepip`, bản bootstrap hiện tại tạo venv bằng `--without-pip`, kiểm tra pip riêng và cài `pip==25.3` bằng `get-pip.py` từ PyPA khi còn thiếu. Cách này không cần apt/sudo hay gói `python3.13-venv` của hệ điều hành. Nó cũng xử lý venv tạo dở có `bin/python` nhưng chưa có pip: chạy lại bước tạo mà không xóa packages/artifacts. Xem [cách cài pip chính thức](https://pip.pypa.io/en/stable/installation/#get-pip-py).

Weights mặc định tải khoảng **29.51 GiB**: teacher BF16 17.98 GiB được quantize NF4 lúc load, student 8.68 GiB, ASR 2.85 GiB. Preflight yêu cầu khoảng **36 GiB scratch trống sau khi cài dependencies** khi cache chưa có weights, cộng dung lượng kit giải nén và outputs/checkpoints. Venv riêng còn cần dung lượng cho CUDA PyTorch và dependencies; lần đầu tải thêm các wheels này. Pip dùng `--no-cache-dir` để tránh giữ thêm bản cache. Đây là dung lượng đĩa, không phải VRAM. Cache weights dùng `/tmp/lifemate_hf`; weights chưa có cache mới được tính vào gate khi chạy lại. Không có cam kết account Kaggle nào cũng đủ scratch hoặc GPU quota.

## 2. Chạy pilot trước

Giữ cell config mặc định:

```python
MODE = 'pilot'
RUN_NAME = 'lifemate_pilot_v6'
DATA_ROOT = ''
INPUT_ARCHIVE = ''
RESUME_FROM = ''
HF_SECRET_NAME = ''
```

Chọn **Run All**. Auto-detect tìm kit trong `/kaggle/input`. Nếu có nhiều bản kit, chỉ rõ một đường dẫn:

```python
DATA_ROOT = '/kaggle/input/<private-dataset>/<folder-chua-train>'
# Hoặc, khi chỉ attach ZIP:
INPUT_ARCHIVE = '/kaggle/input/<private-dataset>/train+public_test.zip'
```

`DATA_ROOT` phải chứa thư mục `train/`. Notebook pin hash manifest train đã audit; kit khác revision sẽ bị từ chối để tránh trộn dữ liệu.

Pilot có 18 stages (gồm stack/precheck và các stages reuse), dùng tối đa 64 fit samples, 8 validation samples, 12 teacher samples, 20 optimizer steps cho mỗi model train. Đây là kiểm tra pipeline có train thật, không phải kết quả full validation. ASR baseline/tuned và agent baseline/tuned được đo trên cùng validation IDs trong run.

Nếu ASR đã hoàn tất trong `lifemate_pilot_v4` và vẫn còn run folder đó trong session hoặc private input, đặt `REUSE_ASR_FROM = '/kaggle/working/lifemate/lifemate_pilot_v4'` trước Run All, dùng RUN_NAME v5 và `RESUME_FROM = ''`. Notebook xác minh source bundle, các hàm ASR, shared core, config ASR, data/runtime/package provenance, split và hashes của completed artifacts rồi copy sang run mới, ghi receipt `asr_reuse.json` và markers có origin `verified_asr_import`. Ba stages ASR được skip, không train lại. Không nhập teacher rollouts thất bại. Nếu kiểm tra không khớp, notebook dừng trước khi copy; không sửa markers/receipt để vượt gate. Khi resume run v5 ở session mới, giữ config cũ; reuse stage đã hoàn tất được kiểm tra checksum và skip, không cần nguồn ASR cũ còn mounted.

Notebook lần lượt:

1. Kiểm tra GPU/packages/model revisions/disk, tạo fixed split và gold references bằng ToolEnv ở job train-only riêng.
2. Đo PhoWhisper gốc, train ASR LoRA, đo lại và chọn ASR.
3. Chạy teacher trên fit subset; lọc bằng state hash + local tool-call matching, dựng next-action examples.
4. Đo student gốc; chạy hai optimizer steps trên example dài nhất để kiểm tra memory thực; train student QLoRA.
5. Đo student bằng gold text và audio thật với ASR đã chọn; xuất adapters/reports.

Student context train mặc định 8192 tokens; prompt/observations bị mask khỏi loss. Example quá dài được ghi vào length report, không cắt target/schema. Coverage dưới 80% sẽ dừng. Loss chỉ tính logits tại vị trí target để giảm memory; vẫn cần pilot xác nhận VRAM thực trên T4.

## 3. Xem kết quả

Run folder:

```text
/kaggle/working/lifemate/lifemate_pilot_v6/
```

Các artifact chính:

| File/folder | Ý nghĩa |
| --- | --- |
| `FINAL_REPORT.json` | Tổng hợp ASR, teacher, student text/audio, memory và parameter gate |
| `asr_adapter/`, `student_adapter/` | LoRA adapters và processor/tokenizer |
| `selected_asr.json` | Dùng ASR gốc hay adapter sau validation |
| `split.json`, `data_report.json` | IDs, nhóm split, phạm vi đánh giá |
| `teacher_metrics.json`, `examples_report.json` | Accepted/rejected teacher samples, số examples teacher thực sự bổ sung |
| `length_report.json`, `student_memory_smoke.json` | Coverage/context và peak memory của smoke test |
| `agent_base_text_metrics.json`, `agent_tuned_text_metrics.json`, `agent_audio_metrics.json` | So sánh student trước/sau và pipeline audio |
| `config.json`, `provenance.json`, `source/`, `logs/`, `stage_*.json` | Config/code/package/data provenance, tiến độ và lỗi |
| `export_manifest.json` | Model revisions, logical parameter counts, artifacts cần đóng gói |

Các metrics là **local diagnostics**, chưa phải điểm BTC. Timing trên Kaggle không chứng minh đạt 30 giây trên máy chấm. Export kiểm tra LLM <5B và tổng inference models <6B bằng logical parameters, nhưng cách tính và rules cuối cùng vẫn cần đối chiếu BTC.

## 4. Chạy full và resume

Sau khi pilot hoàn tất, kiểm tra reports rồi đổi:

```python
MODE = 'full'
RUN_NAME = 'lifemate_full_v1'
RESUME_FROM = ''
```

Full dùng toàn bộ fit/validation đã chia, teacher rollout trên fit và train hai epochs. Không ghi đè pilot. Một notebook có thể cần nhiều GPU sessions; chưa có số đo để cam kết full run hoàn thành trong một phiên Kaggle.

Mặc định `SESSION_BUDGET_HOURS = 8.5`, chừa khoảng dự phòng nếu session chỉ có 10 giờ. Worker dừng tại sample/optimizer boundary trước deadline, lưu checkpoint và notebook tạo:

```text
/kaggle/working/lifemate/<RUN_NAME>_resume.tar.gz
```

Khi dùng **Save Version → Save & Run All**, bật **Save Output** nếu UI có tùy chọn. Đến budget, notebook tự skip các stages còn lại và kết thúc bình thường sau khi ghi archive; thông báo rõ run chưa hoàn tất. Sau khi version xong, kiểm tra tab Output có run folder/archive rồi attach output đó vào phiên sau. Không dựa vào việc Kaggle sẽ giữ được mọi file nếu bị hard timeout/kill bất ngờ. Các checkpoint local không phải bản backup bên ngoài session.

Nếu chạy interactive, **lưu archive hoặc run folder vào storage Private trước khi session hết**. Giữ nguyên notebook/config và data mount path để resume:

- Cùng session: để `RESUME_FROM = ''`, Run All lại. Các completed stages được kiểm tra checksum rồi skip; training tiếp tục từ checkpoint đủ optimizer/trainer state.
- Session mới: attach archive/run folder đã lưu dưới dạng Private, đặt `RESUME_FROM` tới file `*_resume.tar.gz` hoặc folder chứa `config.json`; giữ nguyên `RUN_NAME`, MODE và các config còn lại. Data mount path, code, package versions và loại GPU phải khớp.

Archive chứa outputs/checkpoints, không chứa BTC kit hay base model cache. `/tmp` có thể mất sau session; weights có thể phải tải lại. Chỉ đổi `SESSION_BUDGET_HOURS` để phù hợp giới hạn session thực tế; đổi model/context/code thì dùng run mới.

Nếu session bị kill giữa ToolEnv rollout, `.inflight.json` chưa có committed record sẽ chặn resume để tránh reset state và che calls đã execute. Giữ artifacts để kiểm tra; không xóa guard nhằm chạy lại âm thầm.

## 5. Khi notebook dừng

| Lỗi/gate | Xử lý |
| --- | --- |
| Chỉ có một GPU hoặc teacher OOM | Dùng đúng T4 x2/compute đủ lớn; nếu chủ động đổi teacher thì dùng config và run mới |
| Không đủ scratch | Chuẩn bị disk/cache phù hợp; không bỏ gate khi chưa đo dung lượng thật |
| Student memory smoke OOM | Giữ logs; thay context/LoRA hoặc compute có đo lại coverage, dùng run mới; notebook không tự giảm context |
| Coverage <80% | Xem length report và schemas/history; không hạ gate chỉ để chạy qua |
| Không đủ teacher samples đạt chuẩn | Xem rollout/error metrics; pilot yêu cầu ít nhất 1, full ít nhất 10; không đưa rejected trajectories vào train |
| Teacher `OutOfMemoryError` | Bản mới chia weights bằng `balanced`, budget mặc định 85% free VRAM/GPU và chunked prefill 256 tokens. Xem `teacher_memory_smoke.json` trước rollout; đây là kiểm tra prompt ban đầu dài nhất, không bảo đảm mọi tool history sau đó đều vừa. Nếu vẫn OOM, dừng giữ evidence, cần GPU nhiều VRAM hơn hoặc chủ động chọn teacher nhỏ hơn trong run mới; không hạ quality gate |
| Reference/state consistency lỗi | Kiểm tra đúng kit/runtime revision; không sửa gold/state để vượt gate |
| Package/CUDA/model load lỗi | Lưu log preflight; pinned dependency stack chưa được chứng nhận bằng GPU run thực |

## Phạm vi đã kiểm chứng

Đã có 34 offline tests cho parser, split, source-of-truth ToolEnv logs, permission checks, resume guards, archive safety, session budget, loss/gradient equivalence, bootstrap, lỗi gốc trong runner, warmup arguments, teacher diagnostics/OOM fail-fast, teacher 9B mặc định, reuse ASR và chat EOS. Regression test mô phỏng checkpoint EOS khác chat EOS, xác nhận generator dừng đúng lượt và controller execute đúng một tool call. Bootstrap tests kiểm tra thứ tự cài CUDA PyTorch trước import, constraint giữ đúng CUDA build, môi trường worker không kế thừa hooks, và một venv thật bỏ qua injected `sitecustomize`. Không tải/cài torch trong các tests. Đã kiểm tra round-trip native tool-call/STOP templates với tokenizer thật của student 4B, teacher 9B và teacher 27B trước đó tại pinned revisions: cả 6 cases pass, không tải weights. Notebook được rebuild từ source và kiểm tra syntax/checksum.

Transformers 5.20.2 dùng `warmup_steps=0.05` cho warmup 5%; `warmup_ratio` đã bị bỏ. Đã đối chiếu toàn bộ keywords của `TrainingArguments` trong worker với source chính thức đúng tag 5.20.2. Runner hiện hiển thị marker error, exit code và worker log tail khi stage fail. Nếu cập nhật worker từ bản cũ, dùng RUN_NAME mới để giữ nguyên provenance của run trước. Để xem lỗi trên bản cũ: `print((RUN / 'logs' / 'asr_train.log').read_text(encoding='utf-8', errors='replace')[-16000:])`.

Bộ CUDA PyTorch và wheel Python 3.13 được đối chiếu với [hướng dẫn phiên bản chính thức](https://pytorch.org/get-started/previous-versions/#v291) và [CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/torch/). Chưa đo bootstrap end-to-end trên Kaggle.

**Chưa xác nhận pipeline hoàn tất end-to-end trên Kaggle GPU.** Team đã gửi log chạy ASR/teacher; tôi chưa trực tiếp kiểm chứng memory fix trên GPU đó. Pilot vẫn là bước xác minh trước full run; chưa khẳng định chất lượng teacher/student hay thời gian train.

Sau log Kaggle của team (11/12 teacher rollouts OOM), đã thêm balanced weight placement, chunked prefill, memory smoke trước ToolEnv và dừng ngay sau OOM đã ghi log. Đã chạy kiểm tra CPU với model Qwen3.5 hybrid rất nhỏ, random weights trên Transformers 5.20.2: greedy outputs khi chunked/full prefill giống nhau ở hai độ dài 17/33 tokens. Script tái lập: `oneclick/verify_chunked_prefill.py`. Kết quả này kiểm tra API/cache semantics, không chứng minh teacher 27B vừa VRAM T4 hay bảo đảm mọi prompt đều cho output giống nhau về mặt số học.

Log tiếp theo xác nhận cấu hình 27B balanced vẫn spill sang CPU/disk trước khi load. Bản v4 chủ động đổi teacher sang Qwen3.5-9B, giữ student Qwen3.5-4B và PhoWhisper-medium. Đây là thay đổi model rõ ràng, không phải fallback tự động; chưa có kết quả chất lượng hoặc memory smoke 9B trên Kaggle. Không bật CPU offload để che lỗi memory.

Log v4 tiếp theo: 9B memory smoke pass (7712 prompt tokens, peak khoảng 3.76/4.22 GiB), nhưng 7 timeout và 5 structured-output errors. Đã xác minh metadata thật tại cả ba pinned revisions: checkpoint `text_config.eos_token_id=248044` (`endoftext`) khác tokenizer/chat EOS `248046` (`im_end`). Bản trước chỉ gộp EOS để kiểm tra output sau generation, không truyền chat EOS vào decoder. Bản v5 truyền hợp các EOS trực tiếp vào `generate()`, đồng thời cấu hình generation_config cho cả teacher/student và memory smoke. Giữ model, prompt, token/time budget và quality gate để đo tác động riêng của EOS fix. Dòng log `LLM chat termination` phải hiển thị `generation_eos: [248044, 248046]`. Đây là lỗi code đã xác nhận; cần rerun để đo phần timeout/format errors còn lại, chưa khẳng định mọi sample sẽ pass.

Notebook xuất adapters và reports, chưa tạo Docker hay submission public/private. Offline inference sau này phải đóng gói base weights, adapters, tokenizer/processor và đúng `zaic_tools`; chỉ PhoWhisper + student tham gia inference, teacher không tham gia. Outputs chứa gold/traces của BTC nên luôn giữ Private.

Teacher quality gate: `accepted=0` không đồng nghĩa lỗi load weights. Xem `teacher_metrics.json` diagnostics và các trường status/error/elapsed_s/local_score trong `teacher_rollouts.jsonl` trước khi thay teacher, time budget hoặc threshold. Bản hiện tại giữ nguyên gate, ghi rõ lý do loại và không reload model/replay ToolEnv nếu toàn bộ rollout đã được commit. Không bỏ rejected records hoặc tự chuyển sang gold-only rồi gọi đó là distillation.
