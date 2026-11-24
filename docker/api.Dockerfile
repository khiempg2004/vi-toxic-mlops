FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN pip install uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system ".[api]"

RUN useradd -m -u 1000 app && mkdir -p /app/logs /app/reports && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "vitoxic.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
