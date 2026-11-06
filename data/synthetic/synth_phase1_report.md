# LifeMate — Phase 1 Synthetic Data Report (v2)

> **Trạng thái:** v2 thay thế toàn bộ bản v1 (488 mẫu). Code, split và data đã sinh lại và qua kiểm định; **sẵn sàng bàn giao cho B2** (xem lưu ý sử dụng ở mục 6).
> **Người thực hiện:** [DATA] · **Ngày:** 2026-11-06 · **Root chạy:** `LifeMate/` · **Branch:** `feat/lifemate-data`

---

## 0. Tóm tắt

| Hạng mục | v1 (05/11) | v2 (06/11) |
|---|---|---|
| Chia tập FIT / VAL | 1342 / 203 (VAL 13,1%), **VAL 0% mẫu không có history** | **1237 / 308 (VAL 19,9%)**, phân bố hai phía gần như trùng nhau |
| Số mẫu synthetic đưa vào train | 488 (205 rỗng, 280 hỏi lại, 3 có hành động) | **331** (57 rỗng, 112 hỏi lại, 162 có hành động); pool 496 |
| Tỉ lệ sau khi trộn với FIT (rỗng / hỏi lại / hành động) | 15,5% / 24,4% / 60,0% | **9,3% / 18,2% / 72,5%** (FIT gốc: 7,2% / 14,0% / 78,8%) |
| Validate | "xanh", nhưng không phát hiện được nhãn sai | **0 lỗi / 9 nhóm kiểm tra, replay toàn bộ**; chạy trên data v1 bắt được 473 lỗi |
| Đọc `train_environment/*/state.json` | đọc và copy trực tiếp | **không**, lấy state qua `ToolEnv("train", id).get_state()` |

---

## 1. Lỗi của v1 và cách sửa

| # | Lỗi ở v1 | Hậu quả (đo trên data) | Sửa ở v2 |
|---|---|---|---|
| 1 | Chia tập group theo SHA256 của `history` | 527 mẫu không có history cùng hash `[]` thành **một nhóm**, rơi hết vào FIT. VAL có 0% mẫu không history, chỉ còn 13,1% | Group theo hội thoại (union-find: trùng transcript, trùng hội thoại, tiền tố hội thoại); phân tầng theo hành vi × độ dài history × số call |
| 2 | t2: ID bản sao = `id + "_syn"` | 259/259 mẫu lộ `_syn` trong `tool_results`; train có 0 ID kiểu này | ID theo kiểu train (`c_05` → `c_11`, `chat_03` → `chat_06`) |
| 3 | t2: lấy `writes[0]` | 137/259 mẫu bỏ mất các hành động phía sau; train không có mẫu nào như vậy | Nhắm lệnh ghi **cuối**; replay mọi lệnh gold đứng trước (cả lệnh ghi) |
| 4 | t2: bản sao chép y hệt, số điện thoại xáo chữ số | 229/259 bản sao giống hệt bản gốc; 22/30 số điện thoại không hợp lệ | Đổi thuộc tính **không được nhắc** trong câu (quan hệ trung tính); số mới `09xxxxxxxxx` |
| 5 | t2: nhân bản mọi loại thực thể | 56 mẫu `smart_home` trong khi train chỉ có 1 mẫu hỏi lại về thiết bị | Chỉ sinh cho cặp (tool, tham số x-ref) **đã có** trong các mẫu hỏi lại của FIT, phân bổ theo tỉ lệ train |
| 6 | t3: `re.sub` xoá mọi con số khớp regex | Xoá luôn ngưỡng giá vàng ("hơn thì"), sót "kém 15", không bắt "hai triệu" (nhãn sai ở `t3_00013`) | Chỉ xoá **đúng một cụm** có giá trị bằng giá trị gold (bộ đọc số/giờ tiếng Việt `viet_spans.py`) |
| 7 | t3: không kiểm tra con số thuộc lệnh nào | `t3_00002/04/08`: xoá "50 nghìn" của lệnh nạp tiền nhưng hỏi `amount` của lệnh chuyển; nhãn vẫn `topup_mobile(amount=50000)` | Bỏ mẫu nếu giá trị còn dùng ở lệnh khác, có trong history, hoặc nằm trong nội dung chữ của lệnh khác |
| 8 | t2/t3: `final_state` gán cứng `{}` | Trái quy ước train (79/79 mẫu "có lệnh ghi trước ask" đều có `final_state`) | `final_state` = state các app bị thay đổi, đúng định nghĩa `Executor.replay()` của `zaic_tools` |
| 9 | t4: chèn nhiễu vào bất kỳ token nào, so 1 token với tên đầy đủ | Chỉ ra 3 mẫu; nhiễu rơi vào động từ/hư từ | Chỉ làm nhiễu token **trong tên thực thể** mà lệnh ghi trỏ tới; kiểm tra vẫn đối chiếu được duy nhất |
| 10 | t4: bảng nhầm lẫn dựng trên FIT cũ | Một phần audio nay thuộc VAL → rò thông tin VAL | Dựng lại từ FIT mới (`confusion_fit_v1.json`) |
| 11 | `validate_phase1` ép `final_state == {}`, smoke chỉ xem `ok == true` | Report v1 ghi "replay sạch 0/283" nhưng không kiểm tra nhãn có đúng không | Viết lại: 9 nhóm kiểm tra, replay **toàn bộ** và so từng kết quả (mục 5) |
| 12 | Gộp hết, không giới hạn tỉ lệ | Tỉ lệ rỗng/hỏi lại tăng gấp đôi | `merge_phase1` giới hạn tỉ lệ ≤ 1,3 × FIT (mục 4) |

Một số nhận định trong report v1 không đúng nên đã bỏ:
- "72% mẫu ask phải read để phát hiện trùng tên": con số gốc trong plan là *157/217 mẫu ask đã gọi tool khác trước khi hỏi*. Thực tế train có 131 mẫu ask vì thực thể mơ hồ và 86 mẫu vì thiếu giá trị.
- "Label sinh bằng ToolEnv thật, không bịa": ToolEnv chỉ kiểm tra schema và thực thi, không đảm bảo nhãn đúng ý câu nói (xem lỗi 7).

---

## 2. Chia tập — `split-v1` (`make_fit_manifest.py`)

- **Group (union-find):** R1 trùng transcript hiện tại sau NFC/lowercase; R2 trùng toàn bộ hội thoại; R3 chuỗi câu user của mẫu A là tiền tố history của mẫu B (≥ 2 lượt). Không group theo một câu chung chung trùng nhau (plan 09.4), chỉ báo cáo.
- Kết quả: 1544 group, 1 group 2 mẫu (`nhắc mình gọi lại cho khách lúc 9 giờ kém 15 tối mai nhé.`), còn lại là mẫu đơn. Không có hội thoại nào trải qua hai phía.
- **Phân tầng:** hành vi (rỗng / hỏi lại / hành động) × history (0 / 1–4 / 5+ lượt) × số call (0 / 1 / 2 / 3+), xáo trộn có seed `20261103`, lấy 20% mỗi tầng.

| | FIT (1237) | VAL (308) |
|---|---|---|
| Hành động / hỏi lại / rỗng | 78,8 / 14,0 / 7,2 % | 78,6 / 14,3 / 7,1 % |
| History 0 / 1–4 / 5+ | 34,0 / 36,0 / 30,0 % | 34,4 / 35,7 / 29,9 % |
| Số call 0 / 1 / 2 / 3+ | 11,0 / 24,7 / 26,8 / 37,4 % | 11,4 / 24,4 / 26,6 / 37,7 % |
| Domain chỉ có ở một phía | không | không |
| Near-duplicate (Jaccard ≥ 0,9) VAL↔FIT | | 0 |
| Câu history VAL cũng có trong FIT (chỉ báo cáo) | | 18 |

Đầu ra: `manifests/split_v1.json` (ids, group, seed, sha256 nguồn, quy tắc, thống kê) và `manifests/train_fit_provisional.json` (danh sách id FIT, giữ tên cũ để code khác không phải sửa).

**Lưu ý:** split đổi nên 85/488 mẫu synthetic v1 có nguồn nay thuộc VAL. Toàn bộ synthetic đã sinh lại từ FIT mới.

---

## 3. Synthetic v2 — từng kỹ thuật

Nguyên tắc chung:
- Nguồn chỉ từ TRAIN_FIT (`assert_source`).
- State gốc lấy qua `ToolEnv("train", id).get_state()`; `_meta` của state synthetic dựng lại từ `context`. Đã kiểm chứng trên **1545/1545** mẫu train: cách dựng này cho ra đúng `tool_results` và đúng `final_state` như phiên ToolEnv gốc.
- Seed `20261103`, chạy lại cho kết quả giống hệt (sha256 ở mục 7).

| Kỹ thuật | Pool | Chọn vào merge | Lý do loại chính |
|---|---:|---:|---|
| **t1** gỡ app → `[]` | 191 | 57 | còn app cùng group 495; history đã dùng app 283; còn app thay thế khác group 6 |
| **t2** thực thể trùng tên → hỏi lại | 59 | 59 | cặp (tool, tham số) không có trong train 611; không nhân bản tự nhiên được 60 (thường do câu gọi bằng quan hệ "mẹ", "sếp" chứ không gọi tên) |
| **t3** lược số tiền / giờ → hỏi lại | 84 | 53 | không có tham số lược được 661; giá trị không nằm sẵn trong câu 91; giá trị có trong history 21; câu tự sửa 12 |
| **t4** nhiễu ASR trên tên thực thể | 162 | 162 | lệnh ghi không trỏ tới thực thể có tên trong câu 714; tên nằm trong tham số chấm tuyệt đối 76 |

- **t1:** gỡ mọi app gold dùng; bỏ nếu còn app cùng group hoặc app thay thế khác group (vd `shopping_list` ↔ `notes`/`tasks`); kiểm chứng mọi call gold bị ToolEnv từ chối.
- **t2:** nhắm lệnh ghi cuối; thực thể phải được gọi **đúng tên** trong câu, chưa xuất hiện trong history, không bị lệnh khác dùng. Bản sao có ID kiểu train, số điện thoại mới, quan hệ trung tính nếu câu không nhắc. Replay các lệnh gold đứng trước; kết quả read phải chứa cả hai thực thể. Phân bố: `sms.send` 31, `chat_ott.send_message` 8, `phone.call` 7, `door_lock.grant_guest_code` 3, …
- **t3:** chỉ `amount` (số tiền) và `time`/`start` (giờ). Số tiền thay bằng "chút tiền"/"ít tiền" (kiểu câu train: "chuyển ví cho bố chút tiền"); giờ thì xoá phần giờ, giữ buổi/ngày (train: "nhắc anh uống liều Cardiol sáng mai" vẫn hỏi `time`). Ví dụ: *"…chuyển Lê Minh 2 triệu từ tài khoản thanh toán…"* → *"…chuyển Lê Minh ít tiền từ tài khoản thanh toán…"*; ngưỡng "vượt 123 triệu" giữ nguyên.
- **t4:** đổi đúng 1 token trong tên thực thể theo bảng nhầm lẫn PhoWhisper-small trên FIT (839 cặp, dựng từ 4023/4023 audio của FIT; vd `Minh`→`Mình`, `phòng`→`phần`, `Lan`→`Làng`), bỏ biến thể chính tả (`hoà`/`hòa`). Tên nhiễu (bỏ dấu) phải giống tên đúng nhất và duy nhất (≥ 0,75). Nhãn giữ nguyên gold.

---

## 4. Tỉ lệ trộn (`merge_phase1.py`)

Gold rỗng mà dự đoán có call, hoặc gold có hành động mà dự đoán rỗng/hỏi lại, đều bị 0 điểm. Vì vậy số mẫu t1 và t2+t3 được chọn sao cho sau khi trộn với FIT, tỉ lệ rỗng và tỉ lệ hỏi lại **không vượt 1,3 × FIT**. Tỉ lệ t2 : t3 theo train (mơ hồ thực thể : thiếu giá trị = 97 : 76 trong FIT). t4 lấy hết.

| | Rỗng | Hỏi lại | Hành động |
|---|---:|---:|---:|
| FIT | 7,2% | 14,0% | 78,8% |
| FIT + synthetic v1 (488) | 15,5% | 24,4% | 60,0% |
| **FIT + synthetic v2 (331)** | **9,3%** | **18,2%** | **72,5%** |

Pool đầy đủ vẫn nằm trong `synthetic/phase1_*.jsonl`; đổi hệ số bằng `--cap`.

---

## 5. Kiểm định (`validate_phase1.py` v2)

| Nhóm | Nội dung |
|---|---|
| C1 | Hình dạng nhãn; `ask` là lệnh cuối và duy nhất |
| C2 | Mọi call hợp lệ theo luật `zaic_tools` với `installed_apps` của chính record |
| C3 | Không còn chuỗi `_syn` |
| C4 | Số điện thoại dạng `^0\d{9,10}$` |
| C5 | Nguồn thuộc FIT mới; transcript không trùng câu nào (kể cả history) của VAL |
| C6 | Không trùng (transcript + context) |
| C7 | Replay **toàn bộ**: `tool_results` và `final_state` phải bằng nhãn |
| C8 | State synthetic khớp nguồn (t3/t4 cùng hash với mẫu gốc; t2 = gốc + đúng 1 thực thể) |
| C9 | Ngữ nghĩa theo kỹ thuật (t1 call gold bị từ chối; t2 có ≥ 2 thực thể trùng tên; t3 giá trị đã lược không còn trong câu; t4 đổi đúng 1 token, nhãn = gold) |

Kết quả:
- **v2:** pool 496 và merge 331 → **0 lỗi**.
- **v1** (488, không replay vì không có state cũ): **473 lỗi**: `_syn` 259, số điện thoại 22, nguồn thuộc VAL mới 85, transcript trùng VAL 83, t3 thiếu provenance 21, t4 không phải đổi 1 token 3.
- **Kiểm tra ngược:** cố tình làm hỏng 5 record (xoá `final_state`, trả câu gốc vào t3, sửa kết quả read, thêm `ask` thứ hai, sửa nhãn t4) → bị bắt **5/5**.

---

## 6. Quyết định của DATA và lưu ý khi sử dụng

1. **Dùng synthetic từ B2.** DATA (người phụ trách phần synthetic) quyết định đưa 331 mẫu v2 vào huấn luyện từ B2. Quyết định này thay cho dòng *"không thêm synthetic data trước gate"* trong plan 03/11.
2. **Truy cập state.** Pipeline v2 không đọc file trong `train_environment/`. State lấy qua `get_state()` (README `zaic_tools` cho phép dùng trên train), rồi ghi ra `synthetic/synthetic_environment/` của đội. Code sinh data chỉ chạy offline trên train, không import vào agent.
3. **Audio không khớp text (bên huấn luyện cần biết).** t3 và t4 đổi `audio_text` nhưng `audio` vẫn trỏ tới WAV gốc. Chỉ dùng cho huấn luyện **text** (LLM), không dùng cho ASR hay huấn luyện đầu vào audio. t1 và t2 giữ nguyên câu nên dùng được cả hai.
4. **Hệ số trộn 1,3** là lựa chọn ban đầu, nên đo lại trên VAL ở B3 (đổi bằng `merge_phase1 --cap`).
5. **ASR cho t4:** đã chạy lại `t4a_transcribe` để bổ sung 717 audio của FIT mới (trước thuộc VAL cũ); giờ 4023/4023 audio FIT có transcript.
6. **Chưa giải thích được:** train có vài câu điều kiện mà gold rỗng hoàn toàn (vd *"Nếu giá mua vào vàng SJC hôm nay dưới 129 triệu thì nhắc mình…"*). t1 chưa sinh kiểu này.

---

## 7. Tái lập

Chạy từ `LifeMate/`, venv `.venv_g`:

```
python -m src.lifemate.synthetic.make_fit_manifest
python -m src.lifemate.synthetic.t4b_confusion
python -m src.lifemate.synthetic.run_phase1 --tech t1
python -m src.lifemate.synthetic.run_phase1 --tech t2
python -m src.lifemate.synthetic.run_phase1 --tech t3
python -m src.lifemate.synthetic.run_phase1 --tech t4c
python -m src.lifemate.synthetic.merge_phase1
python -m src.lifemate.synthetic.validate_phase1 --file synthetic/synthetic_phase1.jsonl
```

Chạy thêm `python -m src.lifemate.synthetic.t4a_transcribe` trước `t4b_confusion` nếu `runs/asr_cache/transcripts.jsonl` chưa đủ 4023/4023 audio FIT.

SHA256 (16 ký tự đầu) của lần chạy chốt 06/11 (transcripts đủ 4023/4023):

| File | sha256 |
|---|---|
| `manifests/train_fit_provisional.json` | `1de7723770defd36` |
| `manifests/split_v1.json` | `8c3a5f395fc95de1` |
| `synthetic/phase1_t1.jsonl` | `3478125acd8ca5c5` |
| `synthetic/phase1_t2.jsonl` | `7bb3ec1e12d9ffb8` |
| `synthetic/phase1_t3.jsonl` | `d6905679164d25be` |
| `synthetic/synthetic_phase1.jsonl` | `b27d7e38d70585b7` |

Kết quả chạy chốt: t1 191, t2 59, t3 84, t4 162; merge 331 (t1 57, t2 59, t3 53, t4 162); hai lần validate đều `VALIDATE OK`.

## 8. Giới hạn còn lại

- t2 chỉ 59 mẫu: phần lớn câu gọi người nhận bằng quan hệ ("mẹ", "sếp"), nên không nhân bản theo tên một cách tự nhiên được.
- t3 chỉ xử lý số tiền và giờ; các tham số khác mà train có hỏi (`to`, `date`, `steps`, …) chưa làm.
- Mẫu rỗng chỉ có kiểu "không có app"; train còn các kiểu tự huỷ ("à thôi khỏi"), phủ định, nói chuyện phiếm.
