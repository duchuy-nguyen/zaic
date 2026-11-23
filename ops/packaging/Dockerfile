# Linux AMD64 candidate, NOT yet GPU/portal validated. Build from a private
# staged context with private_assets/, linux_runtime/ and training_code/.
ARG CUDA_BASE=nvidia/cuda:13.2.0-runtime-ubuntu24.04@sha256:3b8afffc99c6ca801e6bf31a426400432cc8ba9dea77bd27ccf8cd67f06424eb
FROM ${CUDA_BASE}
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261103 \
    LD_LIBRARY_PATH=/code/linux_runtime:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /code
# Build-time downloads only. CUDA12.8 wheels run on a compatible CUDA13.2 driver;
# RTX5090 execution and libc/CUDA library compatibility must be tested on GPU.
COPY requirements-linux.txt /code/requirements-linux.txt
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r /code/requirements-linux.txt && pip check
COPY ops/src /code/ops/src
COPY llm/baseline_50_2/baseline50 /code/llm/baseline_50_2/baseline50
COPY private_assets /code/private_assets
COPY linux_runtime /code/linux_runtime
COPY training_code /code/training_code
COPY predict.py predict.sh start_jupyter.sh predict_notebook.ipynb README.md /code/
RUN test -x /code/linux_runtime/llama-server \
    && test -f /code/private_assets/models/asr/model.safetensors \
    && test -f /code/training_code/README.md
CMD ["/bin/bash", "/code/predict.sh"]
