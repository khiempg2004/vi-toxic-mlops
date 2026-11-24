"""Vòng huấn luyện lại: drift? -> gộp dữ liệu mới -> train LoRA -> so sánh -> triển khai nếu tốt hơn.

Chạy: python -m vitoxic.pipeline.retrain --batches-upto 3 [--force] [hydra overrides...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

from vitoxic.config import cfg_path, load_cfg, repo_root
from vitoxic.monitoring import drift
from vitoxic.promote import evaluate_and_promote


def train_in_docker(run_id: str, overrides: list[str]) -> None:
    cmd = [
        "docker", "compose", "--profile", "train", "run", "--rm", "train",
        "python", "-m", "vitoxic.train_lora", f"train.run_id={run_id}", *overrides,
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root())


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches-upto", type=int, required=True, help="Gộp batch_1..N vào tập huấn luyện")
    ap.add_argument("--force", action="store_true", help="Huấn luyện lại kể cả khi không phát hiện drift")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    overrides = [f"data.batches_upto={args.batches_upto}", *args.overrides]
    cfg = load_cfg(overrides)
    summary: dict = {"started_at": datetime.now().isoformat(), "batches_upto": args.batches_upto}

    report = drift.run(cfg)
    summary["drift"] = {k: report.get(k) for k in ("drift_detected", "drift_share", "n_current", "skipped")}
    if not report.get("drift_detected") and not args.force:
        summary["action"] = "no_drift_skip"
        print("Không phát hiện trôi dữ liệu -> không huấn luyện lại")
    else:
        run_id = args.run_id or f"{cfg.model.short}-b{args.batches_upto}-{datetime.now():%Y%m%d-%H%M%S}"
        train_in_docker(run_id, overrides)
        summary["run_id"] = run_id
        summary["promotion"] = evaluate_and_promote(cfg, run_id)
        summary["action"] = "promoted" if summary["promotion"]["promoted"] else "kept_production"

    out = cfg_path(cfg, "reports_dir") / "retrain"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"retrain_{datetime.now():%Y%m%d-%H%M%S}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


if __name__ == "__main__":
    main()
