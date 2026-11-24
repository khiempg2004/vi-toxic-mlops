#!/usr/bin/env bash
# Kịch bản end-to-end đầy đủ vòng đời:
#   dữ liệu -> baseline -> serve -> zero-shot -> LoRA v1 -> promote -> traffic sạch (không drift)
#   -> traffic teencode (drift) -> retrain có dữ liệu mới -> so sánh -> promote nếu tốt hơn.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { echo; echo "================ $* ================"; }

step "1. Cài đặt + test"
make setup
make test

step "2. Dữ liệu (DVC stage prepare)"
make data

step "3. Baseline TF-IDF + LogReg"
make baseline

step "4. Build image & khởi động phục vụ (vLLM GPU${SERVE_GPU:-1} + API + Prometheus + Grafana)"
make build
make serve

step "5. Mốc so sánh: Qwen gốc zero-shot"
make eval-zeroshot

step "6. Huấn luyện LoRA v1 trên train_base (GPU${TRAIN_GPU:-0})"
RUN_V1="qwen1.5b-v1-$(date +%Y%m%d-%H%M%S)"
make train ARGS="train.run_id=${RUN_V1}"

step "7. Đánh giá & triển khai v1"
make promote RUN="${RUN_V1}"
curl -s localhost:8000/model | python3 -m json.tool

step "8. Lưu lượng đợt 1 (dữ liệu sạch) -> kỳ vọng KHÔNG trôi"
: > logs/requests.jsonl
make simulate SOURCE=batch_1 N=2000
make retrain UPTO=1 || true

step "9. Lưu lượng đợt 3 (teencode nặng) -> kỳ vọng PHÁT HIỆN trôi -> huấn luyện lại"
: > logs/requests.jsonl
make simulate SOURCE=batch_3 N=2000
make retrain UPTO=3

step "10. Tổng kết"
curl -s localhost:8000/model | python3 -m json.tool
echo "Báo cáo: reports/  | Grafana: http://<gpu-lab>:3000  | Prometheus: http://<gpu-lab>:9090"
