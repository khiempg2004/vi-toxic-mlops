# Image huấn luyện LoRA (GPU). Torch wheel trên PyPI đã kèm CUDA runtime 12.8.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/cache/huggingface \
    PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && pip install uv

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system ".[train]"

# Mã nguồn được mount vào /workspace lúc chạy (PYTHONPATH ưu tiên src mount).
CMD ["python", "-m", "vitoxic.train_lora"]
