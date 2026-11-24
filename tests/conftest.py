import random

import pandas as pd
import pytest

CLEAN = ["hôm nay trời đẹp quá", "cảm ơn bạn đã chia sẻ", "video hay lắm", "chúc mọi người vui vẻ",
         "mình rất thích bài này", "nội dung bổ ích ghê", "xem xong thấy vui"]
OFFENSIVE = ["vãi cả chưởng luôn", "hay vãi", "vl thật sự", "cái quái gì thế này", "vãi luôn"]
HATE = ["thằng ngu này cút đi", "đồ mất dạy", "lũ đáng ghét", "ngu như bò", "bọn nó là lũ rác rưởi"]


def synthetic_frame(n_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label, pool in [("CLEAN", CLEAN), ("OFFENSIVE", OFFENSIVE), ("HATE", HATE)]:
        for _ in range(n_per_class):
            rows.append({"text": f"{rng.choice(pool)} {rng.choice(['', '!', '...', ' nhé'])}".strip(), "label": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture
def synth_df():
    return synthetic_frame()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Dựng một repo tạm với configs thật + dữ liệu tổng hợp đã xử lý."""
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "configs", tmp_path / "configs")
    monkeypatch.setenv("VITOXIC_ROOT", str(tmp_path))
    proc = tmp_path / "data" / "processed"
    (proc / "stream").mkdir(parents=True)
    for seed, name in enumerate(["train_base", "val", "test", "test_teencode"], start=10):
        synthetic_frame(seed=seed).to_parquet(proc / f"{name}.parquet")
    for i in (1, 2, 3):
        synthetic_frame(seed=i).to_parquet(proc / "stream" / f"batch_{i}.parquet")
    return tmp_path
