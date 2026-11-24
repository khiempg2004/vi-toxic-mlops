"""Tải ViHSD, chuẩn hoá, chia train_base / các đợt dữ liệu đến dần / tập test teencode.

Chạy: python -m vitoxic.data.prepare [hydra overrides...]
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from vitoxic.config import cfg_path, load_cfg
from vitoxic.data.io import split_path
from vitoxic.data.teencode import teencodify
from vitoxic.labels import LABELS, normalize_text

TEXT_COLS = ["free_text", "text", "comment", "content"]
LABEL_COLS = ["label_id", "label", "labels"]


def _map_label(v) -> str:
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        return LABELS[int(v)]
    s = str(v).strip().upper()
    if s.isdigit():
        return LABELS[int(s)]
    if s in LABELS:
        return s
    raise ValueError(f"Nhãn không hợp lệ: {v!r}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_col = next((c for c in TEXT_COLS if c in df.columns), None)
    label_col = next((c for c in LABEL_COLS if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Không tìm thấy cột text/label trong {list(df.columns)}")
    df = df[[text_col, label_col]].dropna()
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str).map(normalize_text),
            "label": df[label_col].map(_map_label),
        }
    )
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def download_raw(cfg: DictConfig) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    raw = cfg_path(cfg, "raw_dir")
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, fname in cfg.data.files.items():
        target = raw / fname
        if not target.exists():
            try:
                hf_hub_download(
                    repo_id=cfg.data.hf_repo,
                    filename=fname,
                    repo_type="dataset",
                    local_dir=raw,
                    token=os.environ.get("HF_TOKEN"),
                )
            except Exception as e:  # noqa: BLE001
                sys.exit(
                    f"Lỗi tải {cfg.data.hf_repo}/{fname}: {e}\n"
                    f"Dataset bị gated: hãy chấp nhận điều khoản tại "
                    f"https://huggingface.co/datasets/{cfg.data.hf_repo} và đặt HF_TOKEN trong .env"
                )
        out[split] = target
    return out


def split_stream(df: pd.DataFrame, base_fraction: float, n_batches: int, seed: int):
    base, rest = train_test_split(df, train_size=base_fraction, stratify=df["label"], random_state=seed)
    rest = rest.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    bounds = np.linspace(0, len(rest), n_batches + 1).astype(int)
    batches = [rest.iloc[bounds[i] : bounds[i + 1]].reset_index(drop=True) for i in range(n_batches)]
    return base.reset_index(drop=True), batches


def apply_teencode(df: pd.DataFrame, prob: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()
    df["text_orig"] = df["text"]
    df["text"] = [teencodify(t, prob, rng) for t in df["text"]]
    df["teencode_prob"] = prob
    return df


def main(overrides: list[str] | None = None) -> dict:
    cfg = load_cfg(overrides)
    files = download_raw(cfg)
    splits = {name: normalize_columns(pd.read_csv(path)) for name, path in files.items()}

    base, batches = split_stream(
        splits["train"], cfg.data.base_fraction, cfg.data.n_batches, cfg.seed
    )
    probs = list(cfg.data.batch_teencode_prob)
    assert len(probs) == cfg.data.n_batches, "batch_teencode_prob phải có n_batches phần tử"

    outputs: dict[str, pd.DataFrame] = {
        "train_base": base,
        "val": splits["val"],
        "test": splits["test"],
        "test_teencode": apply_teencode(splits["test"], cfg.data.test_teencode_prob, cfg.seed + 1000),
    }
    for i, (batch, p) in enumerate(zip(batches, probs, strict=True), start=1):
        outputs[f"batch_{i}"] = apply_teencode(batch, p, cfg.seed + i)

    stats = {}
    for name, frame in outputs.items():
        p = split_path(cfg, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(p, index=False)
        stats[name] = {"n": len(frame), "labels": frame["label"].value_counts().to_dict()}
    stats_path = cfg_path(cfg, "processed_dir") / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main(sys.argv[1:])
