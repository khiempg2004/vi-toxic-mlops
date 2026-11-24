"""Đánh giá một mô hình đang phục vụ trên vLLM (base zero-shot, candidate hoặc production).

Ví dụ:
  python -m vitoxic.evaluate --model base
  python -m vitoxic.evaluate --run-id qwen1.5b-20261124-101500
  python -m vitoxic.evaluate --model production
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
from omegaconf import DictConfig

from vitoxic.client import classify, ensure_adapter_loaded
from vitoxic.config import cfg_path, load_cfg
from vitoxic.data.io import load_split
from vitoxic.metrics import classification_metrics, latency_stats
from vitoxic.registry import adapter_name, candidate_rel_path, get_production


async def _classify_all(texts, base_url, model, concurrency, max_chars, timeout):
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:

        async def one(t):
            async with sem:
                for attempt in range(3):
                    try:
                        return await classify(client, base_url, model, t, max_chars, timeout)
                    except (httpx.TransportError, httpx.HTTPStatusError):
                        if attempt == 2:
                            raise
                        await asyncio.sleep(1 + attempt)

        return await asyncio.gather(*(one(t) for t in texts))


async def _ensure(base_url: str, name: str, path: str):
    async with httpx.AsyncClient() as client:
        await ensure_adapter_loaded(client, base_url, name, path)


def resolve_target(cfg: DictConfig, model: str | None, run_id: str | None) -> tuple[str, str]:
    """Trả về (tên model trên vLLM, nhãn phiên bản). Nạp LoRA nếu cần."""
    vllm = cfg.serve.vllm_url
    if run_id:
        name = adapter_name(run_id)
        asyncio.run(_ensure(vllm, name, f"{cfg.serve.vllm_models_dir}/{candidate_rel_path(run_id)}"))
        return name, run_id
    if model in (None, "base"):
        return cfg.model.name, "base-zeroshot"
    if model == "production":
        prod = get_production(cfg_path(cfg, "models_dir"))
        if prod is None:
            raise SystemExit("Chưa có mô hình production")
        asyncio.run(_ensure(vllm, prod["adapter_name"], f"{cfg.serve.vllm_models_dir}/{prod['adapter_path']}"))
        return prod["adapter_name"], prod["run_id"]
    return model, model


def evaluate_model(cfg: DictConfig, model_name: str, version: str, sets: list[str] | None = None) -> dict:
    sets = list(sets or cfg.eval.sets)
    report: dict = {"model": model_name, "version": version, "sets": {}}
    for s in sets:
        df = load_split(cfg, s)
        if cfg.eval.limit:
            df = df.sample(min(int(cfg.eval.limit), len(df)), random_state=cfg.seed)
        t0 = time.time()
        results = asyncio.run(
            _classify_all(
                list(df["text"]), cfg.serve.vllm_url, model_name, cfg.eval.concurrency,
                cfg.data.max_chars, cfg.serve.timeout_s,
            )
        )
        wall = time.time() - t0
        preds = [r[0] for r in results]
        m = classification_metrics(list(df["label"]), preds)
        m.update(latency_stats([r[2] for r in results]))
        m["throughput_rps"] = len(df) / wall
        report["sets"][s] = m
        print(f"[{version}] {s}: macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
              f"invalid={m['invalid_rate']:.3f} p95={m.get('latency_ms_p95', 0):.0f}ms rps={m['throughput_rps']:.0f}")
    metric = cfg.promote.metric
    report["mean_" + metric] = sum(v[metric] for v in report["sets"].values()) / len(report["sets"])

    out = cfg_path(cfg, "reports_dir") / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{version}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="base | production | <tên model trên vLLM>")
    ap.add_argument("--run-id", default=None, help="run_id của LoRA trong models/candidates")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.overrides)
    name, version = resolve_target(cfg, args.model, args.run_id)
    return evaluate_model(cfg, name, version, args.sets)


if __name__ == "__main__":
    main()
