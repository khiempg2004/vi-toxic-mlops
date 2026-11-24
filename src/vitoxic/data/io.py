from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from vitoxic.config import cfg_path


def split_path(cfg: DictConfig, name: str) -> Path:
    if name.startswith("batch_"):
        return cfg_path(cfg, "stream_dir") / f"{name}.parquet"
    return cfg_path(cfg, "processed_dir") / f"{name}.parquet"


def load_split(cfg: DictConfig, name: str) -> pd.DataFrame:
    p = split_path(cfg, name)
    if not p.exists():
        raise FileNotFoundError(f"Chưa có {p}. Hãy chạy `make data` trước.")
    return pd.read_parquet(p)


def load_training_frame(cfg: DictConfig) -> pd.DataFrame:
    """train_base + các đợt dữ liệu mới đã gán nhãn (1..batches_upto)."""
    frames = [load_split(cfg, "train_base").assign(source="train_base")]
    for i in range(1, int(cfg.data.batches_upto) + 1):
        frames.append(load_split(cfg, f"batch_{i}").assign(source=f"batch_{i}"))
    return pd.concat(frames, ignore_index=True)
