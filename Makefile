SHELL := /bin/bash
-include .env
export

UV      ?= uv
PY      := $(UV) run python
COMPOSE := docker compose
export UID := $(shell id -u)
export GID := $(shell id -g)

.PHONY: help setup data baseline build serve wait down logs train eval-zeroshot evaluate promote \
        simulate drift retrain e2e test lint repro

help:  ## Liệt kê các lệnh
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup:  ## Cài môi trường Python (host, không cần torch) + DVC remote cục bộ
	$(UV) sync --extra api --extra monitor --group dev
	@mkdir -p data models logs reports $${HF_CACHE_DIR:-$$HOME/.cache/huggingface}
	@$(UV) run dvc remote add -f -d --local storage $$HOME/dvc-storage >/dev/null && echo "DVC remote: ~/dvc-storage"

data:  ## Tải + chuẩn bị dữ liệu ViHSD, chia các đợt dữ liệu đến dần
	$(PY) -m vitoxic.data.prepare

baseline:  ## Train TF-IDF + LogReg (mốc so sánh & mô hình dự phòng)
	$(PY) -m vitoxic.baseline

repro:  ## Tái lập data -> baseline -> train bằng DVC
	$(UV) run dvc repro

build:  ## Build image API và image huấn luyện
	$(COMPOSE) build api
	$(COMPOSE) --profile train build train

serve:  ## Khởi động vLLM + API + Prometheus + Grafana
	$(COMPOSE) up -d vllm api prometheus grafana
	@$(MAKE) --no-print-directory wait

wait:
	@echo -n "Chờ API sẵn sàng"; for i in $$(seq 1 120); do \
	  curl -sf localhost:8000/health | grep -q '"vllm":true' && echo " OK" && exit 0; echo -n "."; sleep 5; done; \
	  echo " TIMEOUT"; exit 1

down:  ## Dừng toàn bộ dịch vụ
	$(COMPOSE) --profile train down

logs:
	$(COMPOSE) logs -f --tail=100

train:  ## Huấn luyện LoRA trên GPU (ARGS="train.run_id=v1 data.batches_upto=0")
	$(COMPOSE) --profile train run --rm train python -m vitoxic.train_lora $(ARGS)

eval-zeroshot:  ## Đánh giá Qwen gốc (chỉ dùng câu chỉ dẫn)
	$(PY) -m vitoxic.evaluate --model base

evaluate:  ## Đánh giá LoRA: make evaluate RUN=<run_id>
	$(PY) -m vitoxic.evaluate --run-id $(RUN)

promote:  ## So sánh với production & triển khai nếu tốt hơn: make promote RUN=<run_id>
	$(PY) -m vitoxic.promote --run-id $(RUN)

simulate:  ## Bắn lưu lượng vào API: make simulate SOURCE=batch_3 N=2000
	$(PY) -m vitoxic.simulate --source $(or $(SOURCE),batch_1) --n $(or $(N),2000)

drift:  ## Kiểm tra trôi dữ liệu trên log API gần nhất
	$(PY) -m vitoxic.monitoring.drift

retrain:  ## Vòng huấn luyện lại: make retrain UPTO=3 [FORCE=1]
	$(PY) -m vitoxic.pipeline.retrain --batches-upto $(or $(UPTO),3) $(if $(FORCE),--force,)

e2e:  ## Chạy toàn bộ kịch bản end-to-end
	bash scripts/e2e.sh

test:  ## pytest
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src tests
