from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from vitoxic.labels import LABELS


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str | None]) -> dict:
    """macro-F1 là chỉ số chính. Dự đoán None (không parse được) tính là sai."""
    y_true = list(y_true)
    invalid = sum(p is None for p in y_pred)
    y_pred_filled = [p if p is not None else "__INVALID__" for p in y_pred]
    labels = list(LABELS)
    per_class = f1_score(y_true, y_pred_filled, labels=labels, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred_filled, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred_filled)),
        "f1_per_class": {lab: float(v) for lab, v in zip(labels, per_class, strict=True)},
        "invalid_rate": invalid / max(len(y_true), 1),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred_filled, labels=labels).tolist(),
        },
    }


def latency_stats(latencies_s: Sequence[float]) -> dict:
    if not latencies_s:
        return {}
    arr = np.asarray(latencies_s) * 1000
    return {
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_p99": float(np.percentile(arr, 99)),
        "latency_ms_mean": float(arr.mean()),
    }
