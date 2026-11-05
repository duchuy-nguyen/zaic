# LLM baseline — LifeMate

Folder code LLM nằm ngang hàng với `LifeMate`. Chỉ lưu source code và kiểm thử ở đây; các tài nguyên đã có được dùng lại từ `../LifeMate`.

```text
zaic/
├── llm/
│   ├── local_baseline/          agent, backend, ASR, scorer, runner và tests
│   ├── BASELINE_COPY_MANIFEST.json
│   └── README.md
└── LifeMate/
    ├── data/                    dataset, split, references
    ├── .runtime/                Python ASR, llama.cpp, PhoWhisper
    ├── runs/                    kết quả và checkpoint đã có
    ├── *.gguf                   weights LLM
    └── kaggle/                  notebook/script Kaggle hiện tại
```

## Chạy từ folder llm

Trong PowerShell ở thư mục gốc `zaic`:

```powershell
cd .\llm
```

Xem tham số của runner (không chạy model):

```powershell
python -m local_baseline.experiment --help
```

Chạy kiểm thử CPU bằng Python ASR đã có:

```powershell
..\LifeMate\.runtime\asr-venv\Scripts\python.exe -m unittest discover -s local_baseline -t .
```

Ví dụ chạy Qwen3.5-4B trên 309 VAL có live ASR khi chủ động muốn mở một lượt mới:

```powershell
..\LifeMate\.runtime\asr-venv\Scripts\python.exe -u -m local_baseline.experiment --model qwen35 --mode asr --partition val-main --weights-manifest ..\LifeMate\QWEN35_MODEL_DOWNLOAD_2026-11-04.json
```

Lệnh ví dụ trên chưa được tự chạy khi tạo folder. ASR local hiện tại là PhoWhisper-small. Bản Small/Medium + Qwen trên Kaggle nằm trong notebook riêng, không bị thay đổi bởi việc sắp xếp folder này.

## Những gì được giữ nguyên

- Đây là bản sao code từ `LifeMate/local_baseline`; bản gốc được giữ nguyên để các notebook, scripts và artifacts cũ tiếp tục sử dụng.
- Agent, prompt, backend, scorer, ASR và logic experiment giữ nguyên byte. Chỉ sửa đường dẫn gốc tài nguyên trong `setup_local.py` và working directory của Python child trong các campaign để dùng đúng code ở `llm`.
- Model/dataset/runtime/runs không được copy, di chuyển hoặc nhân đôi. Mặc định chúng vẫn nằm trong `LifeMate`.
- Có thể chỉ định thư mục tài nguyên khác bằng biến môi trường `LIFEMATE_ROOT`; thư mục đó phải có đúng cấu trúc tài nguyên như `LifeMate`.
- `BASELINE_COPY_MANIFEST.json` ghi checksum nguồn/bản sao và các điều chỉnh đường dẫn.
- Tạo folder không khởi chạy benchmark, không fine-tune và không thực hiện Git commit/push.

Không chạy lại một campaign public-test đã hoàn tất hoặc đã bị ngắt chỉ để chọn lại kết quả. Giữ nguyên checkpoint và quy tắc đánh giá đã đóng băng.
