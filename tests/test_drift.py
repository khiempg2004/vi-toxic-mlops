import random

from vitoxic.config import load_cfg
from vitoxic.data.teencode import teencodify
from vitoxic.monitoring import drift

from .conftest import synthetic_frame


def test_features_capture_teencode():
    texts = ["anh không biết em được yêu quá rồi"] * 50
    noisy = [teencodify(t, 1.0, random.Random(i)) for i, t in enumerate(texts)]
    vocab = drift.build_vocab(texts, min_freq=1)
    clean_f, noisy_f = drift.text_features(texts, vocab), drift.text_features(noisy, vocab)
    assert noisy_f["teencode_ratio"].mean() > clean_f["teencode_ratio"].mean()
    assert noisy_f["oov_ratio"].mean() > clean_f["oov_ratio"].mean()
    assert noisy_f["no_diacritic_ratio"].mean() > clean_f["no_diacritic_ratio"].mean()


def test_detect_same_vs_shifted():
    df = synthetic_frame(200, seed=0)
    vocab = drift.build_vocab(df["text"])
    ref = drift.text_features(df["text"][:300], vocab)
    same = drift.text_features(df["text"][300:], vocab)
    shifted = drift.text_features(
        [teencodify(t, 0.9, random.Random(i)) for i, t in enumerate(df["text"][300:])], vocab
    )
    assert not drift.detect(ref, same, 0.01, 0.5)["drift_detected"]
    assert drift.detect(ref, shifted, 0.01, 0.5)["drift_detected"]


def test_run_writes_report(project):
    cfg = load_cfg(["drift.reference_samples=60"])
    texts = [teencodify(t, 0.9, random.Random(i)) for i, t in enumerate(synthetic_frame(80, seed=5)["text"])]
    rep = drift.run(cfg, texts, ["CLEAN"] * len(texts), source="test")
    assert "drift_detected" in rep and (project / "reports" / "drift_latest.json").exists()
