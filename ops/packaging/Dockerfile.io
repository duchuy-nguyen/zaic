ARG PYTHON_BASE=python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
FROM ${PYTHON_BASE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /code
COPY ops/src /code/ops/src
COPY llm/baseline_50_2/baseline50 /code/llm/baseline_50_2/baseline50
COPY ops/packaging/predict.py ops/packaging/predict.sh /code/
ENTRYPOINT ["/bin/sh", "/code/predict.sh"]
CMD ["--check-input"]
