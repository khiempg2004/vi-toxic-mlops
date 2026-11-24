"""Nạp cấu hình Hydra dùng chung cho mọi script (compose API, không đổi cwd)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def repo_root() -> Path:
    env = os.environ.get("VITOXIC_ROOT")
    if env:
        return Path(env).resolve()
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for p in candidates:
        if (p / "configs" / "config.yaml").exists():
            return p
    return Path.cwd()


def load_cfg(overrides: Sequence[str] | None = None) -> DictConfig:
    root = repo_root()
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides or []))
    return cfg


def cfg_path(cfg: DictConfig, key: str) -> Path:
    """Đường dẫn tuyệt đối cho cfg.paths.<key>."""
    p = Path(str(cfg.paths[key]))
    return p if p.is_absolute() else repo_root() / p


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
