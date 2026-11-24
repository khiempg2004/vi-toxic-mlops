"""Cổng triển khai: chỉ đưa candidate lên production nếu tốt hơn bản đang chạy.

Cả hai được đánh giá lại trên CÙNG các tập (gồm cả tập teencode) tại thời điểm so sánh.
Chạy: python -m vitoxic.promote --run-id <run_id> [--force]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from omegaconf import DictConfig

from vitoxic.config import cfg_path, load_cfg
from vitoxic.evaluate import evaluate_model, resolve_target
from vitoxic.registry import get_production, set_production


def decide(candidate_score: float, production_score: float | None, min_delta: float) -> tuple[bool, str]:
    if production_score is None:
        return True, "Chưa có production -> triển khai candidate"
    delta = candidate_score - production_score
    if delta >= min_delta:
        return True, f"Candidate tốt hơn {delta:+.4f} (>= {min_delta})"
    return False, f"Candidate không đủ tốt: {delta:+.4f} (< {min_delta}) -> giữ bản cũ"


def evaluate_and_promote(cfg: DictConfig, run_id: str, force: bool = False) -> dict:
    metric = "mean_" + cfg.promote.metric
    models_dir = cfg_path(cfg, "models_dir")

    name, version = resolve_target(cfg, None, run_id)
    cand = evaluate_model(cfg, name, version)

    prod = get_production(models_dir)
    prod_report = None
    if prod and prod["run_id"] != run_id:
        pname, pversion = resolve_target(cfg, "production", None)
        prod_report = evaluate_model(cfg, pname, pversion)

    ok, reason = decide(cand[metric], prod_report[metric] if prod_report else None, cfg.promote.min_delta)
    if force and not ok:
        ok, reason = True, f"FORCE: {reason}"
    if ok:
        set_production(models_dir, run_id, metrics={metric: cand[metric], **{
            f"{s}_macro_f1": v["macro_f1"] for s, v in cand["sets"].items()}})

    decision = {
        "timestamp": datetime.now(UTC).isoformat(),
        "candidate": run_id,
        "production_before": prod["run_id"] if prod else None,
        "metric": metric,
        "candidate_score": cand[metric],
        "production_score": prod_report[metric] if prod_report else None,
        "promoted": ok,
        "reason": reason,
    }
    out = cfg_path(cfg, "reports_dir") / "promotions"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_id}.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return decision


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    return evaluate_and_promote(load_cfg(args.overrides), args.run_id, args.force)


if __name__ == "__main__":
    main()
