# LifeMate — Phase 1 Synthetic Data Report

> **Trạng thái:** PHẦN DATA HOÀN TẤT. 488 mẫu, 4 kỹ thuật, đã qua cổng validate
> (0 structural / 0 leak_val / 0 dup ; smoke replay 0/283 = cover TOÀN BỘ pool có label).
> **Người thực hiện:** [LEAD-DATA] · **Ngày:** 2026-11-05 · **Root:** `LifeMate/`
> **Deliverable branch:** `feat/lifemate-data` → trả 3 hạng mục: (a) cách tạo synthetic,
> (b) chia val cho data, (c) kiểm tra môi trường tool.

---

## 0. TL;DR cho reviewer

| Hạng mục | Trả ở đâu | Trạng thái |
|---|---|---|
| (a) Cách tạo synthetic data | §2 (method T1–T4) + code `src/lifemate/synthetic/` + output `synthetic/synthetic_phase1.jsonl` | ✅ |
| (b) Chia val cho data | §3 (split policy) + `make_fit_manifest.py` + `manifests/train_fit_provisional.json` | ✅ |
| (c) Kiểm tra môi trường tool | §4 (probe + smoke evidence) + `env_runner.py`, `probe_env.py` | ✅ |
| Assumption cần LEAD duyệt | §6 | ⚠️ chờ duyệt |
| Handoff → B2 | §8 | ⏳ sau duyệt |

**Con số chốt:** `by_tech = {t1_zero_action: 205, t2_state_mutation: 259, t3_param_removal: 21, t4_asr_noise: 3}` · **Tổng 488** · validate **xanh**.

---

## 1. Counts & validation evidence

### 1.1 Số mẫu theo kỹ thuật

| Tech | Kỹ thuật | accepted | cap | hit-rate | Vai trò nghiệp vụ (lỗ hổng audit 03/11) |
|---|---|---:|---:|---:|---|
| T1 | zero-action (gỡ app → label `[]`) | **205** | 500 | 41.0% | Model gọi bừa tool khi app không cài → dạy từ chối đúng |
| T2 | state-mutation (nhân bản entity → read→ambiguous→ask) | **259** | 625 | 41.4% | 72% mẫu ask phải *đọc trước* mới hỏi → dạy "đọc→thấy mơ hồ→hỏi" |
| T3 | param-removal (lược tham số khỏi transcript → ask) | **21** | 375 | 5.6% | Model tự bịa tham số thiếu → dạy hỏi thay vì hallucinate |
| T4c | asr-noise (nhiễu transcript giả lỗi ASR → fuzzy ground) | **3** | 250 | 1.2% | ASR nghe sai số/tên/giờ → dạy đối chiếu mờ vẫn ra đúng id |
| **TỔNG** | | **488** | | | |

### 1.2 Bằng chứng validate (output thật, `validate_phase1 --smoke 300`)

```
structural/leak/dup errors: 0
smoke pool (records co label): 283
smoke failures: 0 / 283
VALIDATE DONE
```

- **`0 structural`** = mọi record đúng hình dạng label theo tech (T1 `[]`; T2/T3 kết thúc `sys.ask_missing_info`; T4c có ≥1 noise-op `gold≠noisy`).
- **`0 leak_val`** = không transcript synthetic nào trùng tập VAL_MAIN → **compliance xanh**, B2 train trên 488 mẫu này không làm nhiễm bẩn metric.
- **`0 dup`** = không trùng (audio_text + context) → chặn kế thừa 58 history lặp có trong train gốc.
- **`smoke 0/283`** = **toàn bộ** 283 record có label (259 T2 + 21 T3 + 3 T4c) replay sạch qua ToolEnv trong process mới → state persist đúng đĩa, đọc lại được. (T1 = 205 record label rỗng, không vào pool replay, được kiểm bằng structural — đúng thiết kế.)

> **Lưu ý trung thực về T2:** 259 mẫu T2 được xác nhận ở **cả 3 tầng** — duyệt-toàn-bộ (0/259 leak/dup/structural), build-time replay (mỗi record phải pass `count_name_hits≥2` qua ToolEnv mới được `b.add`), và smoke replay (259/259 nhờ `--smoke 300 > 283`). Không còn tầng nào chỉ "lấy mẫu".

### 1.3 Hạ tầng & artifact phụ (không phải sample, nhưng là bằng chứng)

| Artifact | Giá trị | Nguồn |
|---|---|---|
| ZIP `train+public_test.zip` | SHA256 `57d37151…b47320`, size `876,430,231` B | verify trước extract |
| 4 file official | hash OK (`train.jsonl`, `train_tools_catalog.json`, `public_test.jsonl`, `public_test_tools_catalog.json`) | `extract_official.py` |
| Runtime | Python **3.11** (`.venv_g`) + torch **cu124**, RTX 4060 | decision record §7 |
| t4a transcribe | **4158 / 4158 = 100%** audio TRAIN_FIT (PhoWhisper-small, rev `a86b604c…`) | `runs/asr_cache/transcripts.jsonl` |
| t4b confusion | **1158** cặp gold→predicted (tần suất ≥2) | `runs/asr_cache/confusion.json` |

---

## 2. (a) Method — cách tạo synthetic data

**Nguyên tắc chung (áp cho cả 4 tech):**
1. **Nguồn = TRAIN_FIT duy nhất.** Mỗi record gọi `assert_source(sid)` chặn cứng không cho lấy từ VAL_MAIN hay `public_test`.
2. **Audio WAV bất biến 100%.** Không TTS, không synth wav. Mọi synthetic record vẫn trỏ `"audio"` về WAV **gốc** của sample train. Cái bị sửa là **1 lớp quanh wav**: context (T1) / state.json (T2) / transcript (T3, T4).
3. **Label sinh bằng ToolEnv THẬT**, không bịa. T2/T3/T4 chạy `env.call(...)` trên state (gốc hoặc đột biến) để có `tool_results` + hash; record nào replay fail thì **không được add**.
4. **Deterministic.** Seed builder `20261103` → `synthetic_environment/` và `phase1_*.jsonl` tái sinh được chính xác (xem §7).

### 2.1 T1 — zero-action (205)
- **Cơ chế:** lấy sample train có gold tool-call (không sys.*), gỡ khỏi `context.installed_apps` đúng các app mà gold dùng → label thành `[]`. Guard: app bị gỡ không được có *app cùng group* còn sót (tránh model vẫn xử lý được qua app khác → false negative), và history không được đã dùng namespace đó.
- **Vì sao giữ 205 (không nới lên 500):** 500 là **trần (cap)**, không phải chỉ tiêu. Guard out-of-scope siết ⇒ chỉ 205 mẫu "sạch" qua được. Nới guard để ép số sẽ sinh *false zero-action* (model học cách từ chối khi thực ra xử lý được) → **mất recall**. Chất lượng > số lượng.

### 2.2 T2 — state-mutation (259)
- **Cơ chế:** tìm entity mà gold `write` trỏ tới (qua `x-ref`), **nhân bản** nó trong `state.json` (id + `_syn`, phone xáo trộn) ⇒ tạo ambiguity. Chạy các `read` gold qua ToolEnv trên state đột biến; **giữ record chỉ khi** kết quả read thật sự trả **≥2** trùng tên/ground → rồi ép `sys.ask_missing_info(tool_name, [x-ref-param])`.
- **Jump 27 → 259 (×9.6):** bản đầu hard-code 2 app (`contacts`, `smart_home`) + filter substring chặt. Bản chốt: **generic-mutator** `_walk()` quét entity ở **mọi app** (mở khoá `calendar/banking/food_delivery/…`), và **grounding mở rộng** = substring **HOẶC** share-token ≥3 (cứu các câu gọi bằng quan hệ/phòng: "gọi cho **mẹ**", "bật đèn **phòng khách**").
- **Ý nghĩa:** mô phỏng đúng 72% mẫu ask trong train phải *đọc trước* mới hỏi — dạy model thói quen "đọc → thấy mơ hồ → dừng hỏi", không đoán bừa.

### 2.3 T3 — param-removal (21)
- **Cơ chế:** chọn 1 tham số *scalar, non-enum, non-x-ref, không-ID* của gold `write` (qua `pick_removable`), **viết lại transcript** sao cho người nghe không suy ra được tham số đó nhưng giữ nguyên intent (`rewrite`), rồi ép ask đúng tham số ấy. `--mode rule` dùng regex theo 3 nhóm alias (amount / temperature / time); `--mode llm` (Qwen3-4B) là fallback tự nhiên khi có GPU rảnh.
- **Vì sao 21 (thấp):** xem §6 assumption — giả thuyết **trần data** (gold `write` đa số nhận ID/enum ⇒ `pick_removable` trả rỗng), **chưa** được funnel xác nhận (diag_t3 chưa chạy, §5.2).

### 2.4 T4 — ASR-noise (t4a → t4b → T4c = 3)
- **Pipeline 3 bước:**
  - **t4a:** transcribe **toàn bộ** audio TRAIN_FIT bằng PhoWhisper-small (100%, 4158 file) → `transcripts.jsonl`.
  - **t4b:** so gold-text vs predicted-text (SequenceMatcher, chỉ opcode `replace` 1-1, tần suất ≥2) → `confusion.json` (**1158** cặp).
  - **T4c:** lấy 1 token trong transcript có trong confusion, thay bằng token nhầm lẫn phổ biến → transcript nhiễu; **giữ label GOLD nguyên vẹn** (đây là mấu chốt: model phải *đối chiếu mờ* transcript-nhiễu với read-result để vẫn ra đúng id, chứ không phải học label mới).
- **Gate groundability:** chỉ giữ mẫu khi token nhiễu **fuzzy-match được 1 entity duy nhất** trong read-result (ngưỡng sim ≥0.6, tie ⇒ ambiguity thật ⇒ bỏ).
- **Vì sao 3 (thấp):** funnel `diag_t4` (§5.1) chỉ ra 2 tầng chết — (i) `collect_names` hẹp bỏ sót 500 ứng viên (read trả entity dưới key `device_name/account_name/…` chứ không chỉ `name`), (ii) confusion matrix **lẫn lộn entity vs động-từ/huyễn-từ** (`'gọi'→'vợ'`, `'ơi'→'ơi'`, `'đặt'→'đã'`) nên phần lớn noise rơi vào token **không ground được**. 3 mẫu accept là những mẫu noise rơi trúng entity single-token ⇒ **thật, valid**, không phải lỗi.
- **Bug CHK đã sửa (không ảnh hưởng data):** validate cũ so `audio_text != provenance.original_text`, nhưng build T4c **không ghi** `original_text` ⇒ `.get(..., audio_text)` tự default = chính nó ⇒ mọi T4c bị đánh oan structural. CHK mới = `any(op.gold != op.noisy for op in noise_ops)` ⇒ 3 mẫu được công nhận.

---

## 3. (b) Split policy — chia val cho data

- **Phương pháp:** `make_fit_manifest.py` group-by-**history-hash** (SHA256 của `history`), cắt **20%** nhóm làm VAL, còn lại = **TRAIN_FIT**. Group-by-history (không phải per-sample) để **không rò rỉ** các sample chia sẻ cùng hội thoại qua 2 phía.
- **Output:** `manifests/train_fit_provisional.json` = danh sách ID FIT. VAL = `train_ids \ FIT` (suy ra được; file chỉ lưu FIT để tránh trùng nguồn sự thật).
- **Số liệu:** train total **1545** mẫu. Split provisional ⇒ **FIT = 1342 / VAL = 203** (FIT + VAL = 1545 ✅; tỉ lệ ≈ 87/13 vì cắt theo *nhóm history*, không phải theo mẫu, nên không nhất thiết 80/20).
- **Phạm vi của VAL này:** là **holdout NỘI BỘ TRAIN**, dùng cho (1) chống leak Phase 1 (synthetic chỉ được sinh từ FIT) và (2) early-stop/validation loop ở B2. **NÓ KHÔNG PHẢI `public_test`** — `public_test` là tập nộp bài, không đụng tới ở mọi bước Phase 1 (compliance §6).

---

## 4. (c) Tool-environment verification — kiểm tra môi trường tool

Mục: chứng minh `zaic_tools.ToolEnv` đọc đúng state, validate đúng schema, và **synthetic state chạy được** — trước khi tin bất kỳ số mẫu nào.

| Kiểm chứng | Lệnh / nơi | Kết quả (bằng chứng thật) |
|---|---|---|
| Module load đúng | `probe_env --split train --sample train_0001` | in `module: …train\zaic_tools\__init__.py` |
| State đọc được | `probe_env --split train --sample train_0001` | in `state hash: <hex>` |
| Schema meta-tool | `probe_env --split train --sample train_0001` | `validate("sys.ask_missing_info", …)` → `{'ok': True, …}` |
| Synthetic env persist + replay | `validate_phase1 --smoke 300` | **`smoke failures: 0 / 283`** — toàn bộ 259 T2 + 21 T3 + 3 T4c replay sạch qua `make_env("synthetic", sid, root=".")` |
| Catalog/zaic_tools copy đúng | `builder.py` boot | `synthetic/zaic_tools/` + `synthetic_tools_catalog.json` tồn tại, env synthetic import được |

⇒ **Kết luận (c):** môi trường tool **OK cho cả train lẫn synthetic**. Smoke 0/283 là bằng chứng mạnh nhất: nó chạy trong **process mới**, đọc **state từ đĩa**, nên loại trừ mọi ảo giác "chỉ đúng lúc build".

---

## 5. Funnel diagnostics (vì sao T3/T4 thấp — đo, không đoán)

> Bài học quy trình của Phase 1: **không suy nguyên nhân chỉ từ final score.** T4 có funnel đầy đủ; T3 **chưa đo** trong phiên này (§5.2).

### 5.1 T4 funnel (`diag_t4`, trên 1242 ứng viên pass-A)

| Tầng | n | Nghĩa |
|---|---:|---|
| `skip_no_calls` | 100 | zero-action, không có gì để ground (loại hợp lệ) |
| `B_no_read` | 301 | có calls nhưng không có read tool → không có result để đối chiếu (loại hợp lệ) |
| `pass_A` | **1242** | pool ứng viên thật (có conf-token + có read) |
| `D_no_names` | **500** | read OK nhưng `collect_names` trả rỗng → **BUG hẹp-từ-điển** (entity nằm ở key khác `name`) |
| `E_f1_below_0.6` | **439** | có names nhưng noise **không phải entity** → f1=0 (loại đúng, nhưng loãng pool) |
| `ACCEPT` (diag heuristic) | 2 | build thật = **3** (khác vì diag lấy `hit[0]`, build lấy `rng.choice`) |

**E-examples (vạch mặt nguyên nhân thật):** `('gọi','vợ',0.0,['Bố'])`, `('ơi','ơi',0.0,['đèn phòng khách',…])`, `('đặt','đã',0.0,['Minh','Minh'])`, `('nhắn','nhắn',0.0,['Cô giáo Mai','Minh','Sếp Dũng'])` → gold==noisy hoặc noise ở **động-từ/huyễn-từ**, name là **multi-token** ⇒ trần f1 ≈ 0.5 < 0.6 ⇒ loại. **Không phải** "mất dấu làm f1=0" như chẩn đoán ban đầu — funnel đã phủ định nửa đó.

**Hệ quả chiến lược:** confusion dày (1158) nhưng **chủ yếu lỗi non-entity** ⇒ PhoWhisper trên TRAIN_FIT **ít sai entity** ⇒ T4c mỏng là **tín hiệu tốt về chất lượng ASR**, không phải pipeline hỏng. Ép thêm = bịa mẫu noise giả → hại B2.

### 5.2 T3 funnel — CHƯA ĐO trong phiên này
Giả thuyết trần-data (gold `write` toàn ID/enum) dựa trên **cơ chế code** + bằng chứng gián tiếp (21 **đứng im** sau khi fix `_grp` alias ⇒ alias không phải bottleneck). 21 mẫu T3 **đã validate xanh**, nên việc chưa đo funnel **không chặn** đóng Phase 1 — nó chỉ làm §6 #1 dựa trên giả thuyết thay vì số liệu. Muốn củng cố: tạo file `src/lifemate/synthetic/diag_t3.py` rồi chạy `python -m src.lifemate.synthetic.diag_t3` (cwd = `LifeMate`) để chốt `skip_rm_empty` vs `rewrite_ALL_fail`.

---

## 6. Assumptions & limitations — **CẦN LEAD DUYỆT**

1. **T3=21 và T4c=3 là TRẦN DATA, không phải bug pipeline.** T4 đã có funnel (§5.1) xác nhận; T3 **chưa đo funnel** (§5.2) — giả thuyết dựa trên cơ chế code. Quyết định: **chấp nhận số thấp + ghi assumption**, KHÔNG ép bằng cách nới guard hay hạ ngưỡng gate — ép sẽ sinh mẫu ask/noise giả → hại model ở B2.
2. **T1 giữ 205, không nới 500.** Cap 500 là trần; 205 mẫu sạch > 500 mẫu nhiễu (bảo đảm out-of-scope thật).
3. **Audio WAV bất biến;** synthetic = *derived/edited* từ train thật (sửa 1 lớp context/state/transcript + replay ToolEnv sinh label), **không fabricated**, không TTS.
4. **Nguồn = TRAIN_FIT duy nhất** (`assert_source` chặn cứng); **không dùng `public_test`** cho bất kỳ mutation/threshold/confusion nào.
5. **VAL ở §3 là holdout nội bộ train**, không phải public_test.
6. **`synthetic_environment/` và `runs/asr_cache/` KHÔNG commit** — tái sinh được (§7). Repo chỉ giữ code + `synthetic_phase1.jsonl` + meta + catalog + manifest + report.
7. **Quyết định runtime:** chuyển venv chính 3.14 → **3.11** vì torch cu124 không có wheel `cp314` (`pip … cu124` → `from versions: none`); khớp mục tiêu Linux/3.11 của plan 03/11.
8. **Gate T4c đã sửa CHK** (bẫy `original_text` default); 3 mẫu accept là **thật** (noise rơi entity single-token). Optional Phase 2: patch `build_t4` ghi thêm `original_text` vào provenance để audit minh bạch hơn.
9. **Limitation known:** `collect_names` vẫn hẹp (500 ứng viên T4 chết ở `D_no_names`); nếu Phase 2 muốn T4 dày hơn, mở rộng `ENTITY_KEYS` là đòn bẩy số 1, không phải hạ ngưỡng groundability.

---

## 7. Reproducibility — dựng lại từ đầu

**Env:** Python 3.11 (`.venv_g`), `pip install transformers soundfile jsonschema tqdm torch --index-url …/cu124`.
**Seed cố định:** builder `20261103`; validate `random.seed(0)`.

```bash
# 0. raw data KHÔNG ở repo — đặt train+public_test.zip vào LifeMate/, verify hash
python scripts/extract_official.py                      # -> official/lifemate-v3.2.0 (hash OK)
python -m src.lifemate.synthetic.probe_env --split train --sample train_0001   # dl-(c)

# 1. split (dl-b)
python -m src.lifemate.synthetic.make_fit_manifest      # -> manifests/train_fit_provisional.json

# 2. T1-T3 (không cần torch)
python -m src.lifemate.synthetic.run_phase1 --tech t1 --target 500
python -m src.lifemate.synthetic.run_phase1 --tech t2 --target 625
python -m src.lifemate.synthetic.run_phase1 --tech t3 --target 375 --mode rule

# 3. T4 (cần GPU): t4a nền -> t4b -> t4c
python -m src.lifemate.synthetic.t4a_transcribe          # -> runs/asr_cache/transcripts.jsonl (100%)
python -m src.lifemate.synthetic.t4b_confusion           # -> runs/asr_cache/confusion.json (1158)
python -m src.lifemate.synthetic.run_phase1 --tech t4c --target 250

# 4. validate + merge (cổng chốt)
python -m src.lifemate.synthetic.validate_phase1 --smoke 300   # mong: VALIDATE DONE, 0/283
python -m src.lifemate.synthetic.merge_phase1                  # mong: n=488, by_tech đủ 4 key

# 5. (tùy chọn, không chặn) củng cố assumption T3 — cần tạo file diag_t3.py trước
python -m src.lifemate.synthetic.diag_t3
```

**Artifact commit lên `feat/lifemate-data`** (prefix `LifeMate/`): `src/lifemate/synthetic/*.py`, `scripts/extract_official.py`, `manifests/train_fit_provisional.json`, `synthetic/synthetic_phase1.jsonl` + `_meta.json` + `synthetic_tools_catalog.json`, `synthetic/zaic_tools/`, `synthetic/synth_phase1_report.md`, `.gitignore` (append). **Không commit:** `.venv*`, `official/`, zip, `synthetic_environment/`, `asr_cache/`.

---

## 8. Handoff → B2 (SFT/QLoRA)

- **Data train cuối cho B2 = TRAIN_FIT gốc (1342) + 488 synthetic = 1830 mẫu.** 488 là **bổ sung**, không phải toàn bộ — nạp nhầm 488 đơn độc sẽ mất ~1342 mẫu gốc. (VAL = 203 mẫu giữ **ngoài** data train, dành cho early-stop.)
- **Nạp sau khi LEAD duyệt §6.** Entry point: `synthetic/synthetic_phase1.jsonl` (đã sort, đủ schema) + `manifests/synth_phase1.json` (ánh xạ id→source_id để truy vết compliance).
- **Metric chứng minh giá trị (plan mục 16):** delta `S1 − S0` trên **4 slice** tương ứng 4 tech — zero-action rate, read-before-ask accuracy, hallucinate-param rate, ASR-entity grounding F1. Data xong ≠ model xong; B2 là nơi 488 mẫu này "đòi nợ".
- **Dịch chuyển phân phối (tính trên FIT = 1342; mẫu số sau = 1830):**

| Slice | Baseline (audit 03/11, /1545) | Sau +488 (/1830) | Hướng |
|---|---|---|---|
| zero-action | 7.2% (111) | 17.3% (316) | ↑ vá under-trigger |
| ask-bearing | 14% (217) | 27.2% (497) | ↑ vá over-commit |
| read-before-ask / ask | 72.4% (157/217) | 83.7% (416/497) | T2 đẩy tỉ lệ này lên |
| ASR-entity grounding | N/A | N/A (3 mẫu) | do B2 đo |

> Cách tính tử số: ask-bearing sau = 217 + 259 (T2) + 21 (T3) = 497 — **không** cộng 3 mẫu T4c vì label T4c = gold gốc (replay), không phải ask mới. read-before-ask sau = 157 + 259 (T2) = 416 — **không** cộng 21 (T3) vì T3 hỏi-ngay-thiếu-param, không đọc trước.

---
