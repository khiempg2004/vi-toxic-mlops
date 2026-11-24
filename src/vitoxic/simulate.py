"""Mô phỏng lưu lượng thật: bắn bình luận từ một split vào API.

Chạy: python -m vitoxic.simulate --source batch_3 --n 2000
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from vitoxic.config import load_cfg
from vitoxic.data.io import load_split


async def send(api_url: str, texts: list[str], batch_size: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    counts: dict[str, int] = {}
    errors = 0
    async with httpx.AsyncClient(timeout=120) as client:

        async def one(chunk):
            nonlocal errors
            async with sem:
                try:
                    r = await client.post(f"{api_url}/predict/batch", json={"texts": chunk})
                    r.raise_for_status()
                    for p in r.json():
                        counts[p["label"]] = counts.get(p["label"], 0) + 1
                except httpx.HTTPError:
                    errors += len(chunk)

        chunks = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        await asyncio.gather(*(one(c) for c in chunks))
    return {"sent": len(texts), "errors": errors, "labels": counts}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="batch_1")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.overrides)
    df = load_split(cfg, args.source)
    df = df.sample(min(args.n, len(df)), random_state=int(time.time()) % 10_000)
    t0 = time.time()
    res = asyncio.run(send(cfg.serve.api_url, df["text"].tolist(), args.batch_size, args.concurrency))
    res["seconds"] = round(time.time() - t0, 1)
    res["rps"] = round(res["sent"] / max(res["seconds"], 1e-6), 1)
    print(res)
    return res


if __name__ == "__main__":
    main()
