#!/usr/bin/env bash
# Đồng bộ mã nguồn lên máy GPU (không gồm dữ liệu/mô hình/secret).
set -euo pipefail
HOST="${1:-ubuntu@gpu-lab}"
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude /.venv --exclude /data --exclude /models --exclude /logs --exclude /reports \
  --exclude /.env --exclude /wandb --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache \
  ./ "$HOST":~/vi-toxic-mlops/
