"""Mốc so sánh TF-IDF + Logistic Regression (CPU). Cũng là mô hình dự phòng của API.

Chạy: python -m vitoxic.baseline [hydra overrides...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from vitoxic.config import cfg_path, load_cfg
from vitoxic.data.io import load_split, load_training_frame
from vitoxic.metrics import classification_metrics


def build_pipeline(C: float = 4.0, max_iter: int = 2000, min_df: int = 2) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, max_features=200_000)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True, max_features=300_000
                ),
            ),
        ]
    )
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def train_and_eval(train_texts, train_labels, eval_sets: dict, C: float, max_iter: int, min_df: int = 2):
    pipe = build_pipeline(C, max_iter, min_df)
    t0 = time.time()
    pipe.fit(list(train_texts), list(train_labels))
    results = {"train_seconds": time.time() - t0, "sets": {}}
    for name, (texts, labels) in eval_sets.items():
        preds = pipe.predict(list(texts))
        results["sets"][name] = classification_metrics(list(labels), list(preds))
    return pipe, results


def main(overrides: list[str] | None = None) -> dict:
    cfg = load_cfg(overrides)
    train_df = load_training_frame(cfg)
    eval_sets = {}
    for name in ["val", *cfg.eval.sets]:
        df = load_split(cfg, name)
        eval_sets[name] = (df["text"], df["label"])

    pipe, results = train_and_eval(
        train_df["text"], train_df["label"], eval_sets, cfg.baseline.C, cfg.baseline.max_iter
    )
    out_dir = cfg_path(cfg, "models_dir") / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")

    reports = cfg_path(cfg, "reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "tfidf-logreg",
        "train_size": len(train_df),
        **{f"{k}_macro_f1": v["macro_f1"] for k, v in results["sets"].items()},
    }
    Path(reports / "baseline_metrics.json").write_text(json.dumps(summary, indent=2))
    Path(reports / "baseline_full.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
