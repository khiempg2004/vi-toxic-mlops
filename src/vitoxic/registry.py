"""Model registry tối giản dựa trên file.

models/registry/production.json trỏ tới bộ LoRA đang chạy. API đọc file này
(theo mtime) nên chỉ cần ghi đè là chuyển phiên bản, không phải khởi động lại.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def adapter_name(run_id: str) -> str:
    return f"vitoxic-{run_id}"


def candidate_rel_path(run_id: str) -> str:
    return f"candidates/{run_id}"


def _registry_dir(models_dir: Path) -> Path:
    return Path(models_dir) / "registry"


def production_file(models_dir: Path) -> Path:
    return _registry_dir(models_dir) / "production.json"


def get_production(models_dir: Path) -> dict | None:
    f = production_file(models_dir)
    if not f.exists():
        return None
    return json.loads(f.read_text())


def set_production(models_dir: Path, run_id: str, metrics: dict | None = None, extra: dict | None = None) -> dict:
    reg = _registry_dir(models_dir)
    reg.mkdir(parents=True, exist_ok=True)
    previous = get_production(models_dir)
    entry = {
        "run_id": run_id,
        "adapter_name": adapter_name(run_id),
        "adapter_path": candidate_rel_path(run_id),
        "metrics": metrics or {},
        "promoted_at": datetime.now(UTC).isoformat(),
        "previous_run_id": previous["run_id"] if previous else None,
        **(extra or {}),
    }
    # Ghi nguyên tử để API không bao giờ đọc phải file dở dang.
    fd, tmp = tempfile.mkstemp(dir=reg, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(entry, fh, indent=2, ensure_ascii=False)
    os.chmod(tmp, 0o644)
    os.replace(tmp, production_file(models_dir))
    with open(reg / "history.jsonl", "a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
