# Release synthetic_phase1_v21 (tự sinh bởi release_phase1.py — không sửa tay)

| Kỹ thuật | Pool | Trong bản gộp |
|---|---:|---:|
| t1 không có app → [] | 191 | 78 |
| t2 trùng tên → hỏi lại | 23 | 23 |
| t3 thiếu số tiền/giờ → hỏi lại | 84 | 84 |
| t4 nhiễu ASR tên thực thể | 367 | 367 |
| **Tổng** | **665** | **552** |

| | Rỗng | Hỏi lại | Hành động |
|---|---:|---:|---:|
| TRAIN_FIT | 7.2% | 14.0% | 78.8% |
| TRAIN_FIT + synthetic v2.1 | 9.3% | 15.7% | 75.0% |

- Validate: **0 lỗi**, 11 nhóm kiểm tra, replay all_records_with_calls.
- Mẫu chỉ train text (t3, t4): 451.
- dataset sha256: `968a2f3153f0e39ece215d816b12c3178cf67c1dac1e9e54b045a1513939d062`
- split manifest sha256: `8c3a5f395fc95de1813483b42768bd7d29a9df691d67f5899708e4c509ce6a5e`
- confusion sha256: `c3a4fee8ea708d9c991870987b213507adedb93ca6bf59e1e8cb2ba8fe658e7a`
