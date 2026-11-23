# Phân xử của DATA — synthetic phase1 v2.1

## synth_v21_t2_00016 — **SAI NHÃN (xác nhận)**

| | |
|---|---|
| Câu | "Mai Anh **nhà chị** mới đổi email rồi em, sửa trong danh bạ dùm chị…" |
| Bản gốc / bản sao | Mai Anh — *em gái* / Mai Anh — *bạn cùng lớp* |
| Nhãn sinh ra | `sys.ask_missing_info(contacts.update, [contact_id])` |
| Kết luận | "nhà chị" chỉ người trong gia đình → câu đã chọn được *em gái*; hỏi lại là sai. |

Hệ quả:
- Release OPS `synthetic_phase1_v21_text_551` (= v2.1 bỏ đúng ID này) là **bản v2.1 dùng để train text**. `confirmed_label_error` của ID này = **True**.
- DATA **không sinh lại** v2.1 để giữ nguyên ID/hash mà OPS và LLM đã pin.
- Generator đã thêm luật `family_context_near_name` (`mutators.py`): bỏ mẫu khi "nhà" đứng ngay trước hoặc sau tên. Chạy lại t2 với luật mới chỉ loại đúng mẫu này (22/23 giữ nguyên nguồn), nên áp dụng được cho các phiên bản sau.

## Các mẫu còn lại

`audit_phase1.csv` (167 dòng: toàn bộ t2, t3 + 40 t4 + 20 t1) vẫn **chờ người duyệt**. Mẫu nào bị đánh SAI thì ghi thêm vào file này và vào `excluded_samples.csv` của release kế tiếp.
