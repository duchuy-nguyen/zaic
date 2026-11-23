# LifeMate — Phase 1 Synthetic Data Report (v2.1)

> **Trạng thái:** v2.1 thay thế v2 (331 mẫu) và bản OPS `synthetic_phase1_v2_filtered_311`. ID mới có tiền tố `synth_v21_`, không trùng ID cũ.
> **Số liệu chốt** (số mẫu, tỉ lệ, hash) nằm trong `release_v21.md` / `release_v21.json`, do `release_phase1.py` tự sinh. Report này không chép tay các con số có thể đổi.
> **Người thực hiện:** [DATA] · **Ngày:** 2026-11-07 · **Root chạy:** `LifeMate/`

---

## 0. Trả lời review OPS (07/11)

| Điểm OPS nêu | Xử lý trong v2.1 |
|---|---|
| `synth_t2_00014` — "partner Kevin Park" tự phân biệt được người nhận, nhãn vẫn hỏi `contact_id` | Bỏ mẫu khi ngay **trước** tên có từ chỉ vai trò (partner, thầy, sếp, khách hàng, …) |
| `synth_t2_00047` — "Toàn trực ca" khớp với quan hệ "quản lý trực ca" | Bỏ mẫu khi từ **quanh** tên (2 từ trước, 3 từ sau, trừ đại từ xưng hô) trùng với quan hệ của bản gốc hoặc bản sao |
| 18 bản sao t2 chỉ khác ID | Tắt hẳn kiểu bản sao này (kể cả chat). Ghi chú: train **có** chat trùng tên giống hệt nhau trừ ID (`chat_01`/`chat_02` "Minh"); DATA vẫn chọn không sinh kiểu này. Bật lại ở `mutators.IDENTICAL_OK_LISTS` nếu cần |
| Bản 311 vẫn còn t2 tên kiểu danh xưng | Kiểm tra mới C11 bắt **24/39** mẫu t2 trong bản 311 có tên như "Mẹ", "Bố", "Sếp Dũng", "Chị Ly". v2.1 chỉ nhân đôi **tên riêng trơn**, đúng kiểu train (Minh 28, Lan 27, Hương 8, Bình, Trang) |
| Chưa kiểm: `official_train_source_state_provenance` | Đã có ở **C8**: t3/t4 cùng hash state với mẫu train gốc; t2 = state gốc + đúng 1 thực thể |
| Chưa kiểm: `original_train_VAL_text_leakage` | Đã có ở **C5**: nguồn thuộc FIT; transcript không trùng câu nào (kể cả history) của VAL |
| Chưa kiểm: `audio_text_alignment` | Thêm **C10**: t1/t2 `audio_text` và `audio` trùng mẫu gốc (WAV khớp text); t3/t4 khác câu gốc và được đánh dấu chỉ train text |
| Chưa kiểm: `full_independent_semantic_adjudication` | Máy không chứng nhận được. Thêm `audit_phase1.csv`: **toàn bộ** t2 + t3, 40 mẫu t4, 20 mẫu t1, mỗi dòng có câu hỏi kiểm tra để người duyệt điền OK / SAI / NGHI. `audit_phase1 --apply` in ra danh sách ID SAI để loại |
| Chưa kiểm: `training_quality_on_VAL` | Thuộc phần huấn luyện (B2/B3) |

`validation_v21.json` lưu kết quả 11 nhóm kiểm tra để OPS đối chiếu.

---

## 1. Thay đổi v2 → v2.1

| Kỹ thuật | v2 | v2.1 |
|---|---|---|
| **t2** | 59 mẫu, có tên danh xưng, có bản sao chỉ khác ID, chưa chặn từ chỉ vai trò | chỉ tên riêng trơn (người) hoặc thực thể khác ở tác giả/nhà cung cấp/giáo viên; bản sao người khác **id + số điện thoại + quan hệ**; chặn từ chỉ vai trò trước/quanh tên |
| **t4** | 1 biến thể / câu nguồn, chỉ thử 1 thực thể | thử mọi thực thể có tên trong câu; tối đa **2 biến thể / câu nguồn**, mỗi biến thể đổi một token gốc khác nhau |
| t1, t3 | — | giữ nguyên logic |
| ID | `synth_t2_00014` … | `synth_v21_t2_00001` … |
| Validate | C1–C9 | thêm **C10** (khớp audio–text), **C11** (t2 không chỉ-khác-ID, tên là tên riêng), xuất `--json-out` |
| Bàn giao | merge + report | thêm `audit_phase1.csv`, `release_v21.json`, `release_v21.md` |

t2 giảm số mẫu là chấp nhận được: ưu tiên nhãn đúng hơn số lượng. Phần "sinh thêm" lấy từ t4 (mẫu có hành động thật). Vì tỉ lệ rỗng/hỏi lại bị giới hạn theo tổng số mẫu, có thêm t4 thì t1 cũng được nhận thêm.

---

## 2. Chia tập — `split-v1` (không đổi so với v2)

FIT **1237** / VAL **308** (19,9%), group theo hội thoại, phân tầng theo hành vi × history × số call; hai phía có phân bố gần như trùng nhau. Chi tiết và thống kê ở `manifests/split_v1.json` (sha256 `8c3a5f39…`). Nhóm huấn luyện pin đúng manifest này.

---

## 3. Kỹ thuật (v2.1)

Nguyên tắc chung:
- Nguồn chỉ từ TRAIN_FIT.
- State gốc lấy qua `ToolEnv("train", id).get_state()`, không đọc file `train_environment/`; `_meta` dựng lại từ `context`. Đã kiểm chứng 1545/1545 mẫu train cho ra đúng `tool_results` và `final_state`.
- Seed cố định, chạy lại cho kết quả giống hệt.

- **t1 — gỡ app → `[]`:** gỡ mọi app gold dùng; bỏ nếu còn app cùng group hoặc app thay thế khác group; mọi call gold phải bị ToolEnv từ chối.
- **t2 — trùng tên → hỏi lại:**
  - Chỉ dùng cặp (tool, tham số x-ref) có trong các mẫu hỏi lại của FIT, cộng `chat_ott.send_location` và `chat_ott.call`.
  - Nhắm lệnh ghi cuối; thực thể phải được gọi đúng tên, chưa có trong history, không bị lệnh khác dùng.
  - **Người:** tên riêng trơn; bản sao đổi ID theo kiểu train, số `09xxxxxxxxx`, quan hệ trung tính (đồng nghiệp, khách hàng, bạn, hàng xóm, …). Bỏ nếu câu đã nêu quan hệ hoặc có từ chỉ vai trò quanh tên.
  - **Thực thể khác:** bản sao lấy tác giả / nhà cung cấp / giáo viên của một thực thể khác cùng danh sách. Không có thì bỏ (máy cùng phòng, bài hát đã nêu ca sĩ, cuộc họp, khóa cửa).
- **t3 — thiếu số tiền/giờ → hỏi lại:** chỉ xóa đúng một cụm có giá trị bằng giá trị gold. Số tiền thay bằng "chút tiền"/"ít tiền"; giờ thì xóa phần giờ, giữ buổi/ngày. Bỏ mẫu tự sửa, giá trị còn dùng ở lệnh khác, nằm trong history, hoặc nằm trong nội dung chữ của lệnh khác. Đã thử mở rộng sang tham số số nguyên khác (`steps`, `term_months`, …): chỉ thêm khoảng 13 mẫu và nhiều mẫu có giá trị suy ra từ ngữ cảnh, nên **không làm**.
- **t4 — nhiễu ASR trên tên thực thể:** đổi 1 token trong tên theo bảng nhầm lẫn PhoWhisper-small dựng từ FIT; bỏ biến thể chính tả (`hoà`/`hòa`). Tên nhiễu (bỏ dấu) phải giống tên đúng nhất và duy nhất (≥ 0,75). Bỏ nếu tên nằm trong tham số chấm tuyệt đối. Nhãn giữ nguyên gold.

---

## 4. Tỉ lệ trộn

`merge_phase1` chọn số mẫu t1 (rỗng) và t2+t3 (hỏi lại) sao cho sau khi trộn với FIT, tỉ lệ rỗng và hỏi lại **≤ 1,3 × FIT**; t4 lấy hết. Bảng số liệu thực tế ở `release_v21.md`. Hệ số 1,3 là giá trị khởi đầu; đổi bằng `--cap` sau khi đo trên VAL.

---

## 5. Kiểm định

Có 11 nhóm kiểm tra (C1–C11, chi tiết trong docstring `validate_phase1.py`); replay **toàn bộ** mẫu có call.

| Dataset | Kết quả |
|---|---|
| v2.1 (pool và bản gộp) | 0 lỗi |
| OPS filtered-311 | C11: 24 mẫu t2 tên kiểu danh xưng |
| v1 (488) | 732 lỗi (`_syn` 259, bản sao chỉ-khác-ID 259, số điện thoại 22, nguồn thuộc VAL 85, …) |
| Kiểm tra ngược (cố tình làm hỏng 7 record, kể cả bản sao chỉ-khác-ID và t4 không nhiễu) | bắt 7/7 |

**Giới hạn:** validate chứng minh nhãn **chạy đúng và nhất quán**, không chứng minh nhãn **đúng ý người nói**. Phần đó dựa vào `audit_phase1.csv` (người duyệt).

---

## 6. Quyết định của DATA và lưu ý khi sử dụng

1. DATA quyết định dùng synthetic từ B2, thay cho dòng "không thêm synthetic data trước gate" trong plan 03/11.
2. **t3/t4 chỉ train text** (`audio` vẫn là WAV gốc). t1/t2 giữ nguyên câu, dùng được cả text lẫn audio.
3. Dùng bản gộp `synthetic_phase1.jsonl` **bổ sung** vào TRAIN_FIT, không thay thế FIT. Không trộn với ID của bản v2/311.
4. Data (`.jsonl`, `synthetic_environment/`) để trên Drive `ZAIC/data/synthetic_phase1_v21`. Trên git chỉ để code, manifest ID, report và metadata release.
5. Chưa giải thích được: vài câu điều kiện trong train có gold rỗng hoàn toàn; t1 chưa sinh kiểu này.

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
python -m src.lifemate.synthetic.validate_phase1
python -m src.lifemate.synthetic.validate_phase1 --file synthetic/synthetic_phase1.jsonl --json-out synthetic/validation_v21.json
python -m src.lifemate.synthetic.audit_phase1
python -m src.lifemate.synthetic.release_phase1
```

Bước `release_phase1` từ chối chạy nếu validate còn lỗi hoặc hash không khớp.

## 8. Giới hạn còn lại

- t2 ít mẫu: phần lớn câu gọi người nhận bằng quan hệ hoặc danh xưng, không nhân bản tự nhiên được.
- t4 có 2 biến thể / câu nguồn: cùng nhãn, khác một token, nên nếu muốn tách VAL nội bộ cho synthetic thì tách theo `source_id`.
- Mẫu rỗng chỉ có kiểu "không có app"; train còn các kiểu tự hủy, phủ định, nói chuyện phiếm.
