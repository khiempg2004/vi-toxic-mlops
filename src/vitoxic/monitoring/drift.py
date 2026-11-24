"""Phát hiện trôi dữ liệu: so sánh bình luận gần đây (log API) với dữ liệu huấn luyện.

Đặc trưng văn bản: độ dài, tỉ lệ từ không dấu, tỉ lệ token teencode, tỉ lệ từ ngoài từ vựng.
Kiểm định KS từng đặc trưng; trôi nếu tỉ lệ đặc trưng bị trôi >= share_threshold.
Kèm báo cáo HTML của Evidently (nếu cài).

Chạy: python -m vitoxic.monitoring.drift [--current-split batch_3] [hydra overrides...]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from vitoxic.config import cfg_path, load_cfg
from vitoxic.data.io import load_split
from vitoxic.data.teencode import TEENCODE_TOKENS
from vitoxic.labels import LABELS

WORD_RE = re.compile(r"\w+", re.UNICODE)
FEATURES = ["n_chars", "n_words", "no_diacritic_ratio", "teencode_ratio", "oov_ratio"]


def tokenize(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(str(text).lower()) if not w.isdigit()]


def build_vocab(texts, min_freq: int = 2) -> set[str]:
    c = Counter(w for t in texts for w in tokenize(t))
    return {w for w, n in c.items() if n >= min_freq}


def text_features(texts, vocab: set[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        words = tokenize(t)
        n = max(len(words), 1)
        rows.append(
            {
                "n_chars": len(str(t)),
                "n_words": len(words),
                "no_diacritic_ratio": sum(w.isascii() for w in words) / n,
                "teencode_ratio": sum(w in TEENCODE_TOKENS for w in words) / n,
                "oov_ratio": sum(w not in vocab for w in words) / n,
            }
        )
    return pd.DataFrame(rows, columns=FEATURES)


def detect(ref_feats: pd.DataFrame, cur_feats: pd.DataFrame, p_value: float, share_threshold: float) -> dict:
    features = {}
    for f in FEATURES:
        res = ks_2samp(ref_feats[f], cur_feats[f])
        features[f] = {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "drifted": bool(res.pvalue < p_value),
            "reference_mean": float(ref_feats[f].mean()),
            "current_mean": float(cur_feats[f].mean()),
        }
    share = sum(v["drifted"] for v in features.values()) / len(features)
    return {"drift_share": share, "drift_detected": share >= share_threshold, "features": features}


def label_shift(ref_labels, cur_labels) -> dict:
    ref = pd.Series(ref_labels).value_counts(normalize=True).reindex(LABELS, fill_value=0)
    cur = pd.Series(cur_labels).value_counts().reindex(LABELS, fill_value=0)
    expected = np.maximum(ref.values * cur.sum(), 1e-6)
    expected = expected * cur.sum() / expected.sum()
    res = chisquare(cur.values, expected)
    return {
        "reference_dist": ref.round(4).to_dict(),
        "current_dist": (cur / max(cur.sum(), 1)).round(4).to_dict(),
        "p_value": float(res.pvalue),
    }


def read_recent_logs(path: Path, window: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["text", "label"])
    with open(path, encoding="utf-8") as fh:
        lines = deque(fh, maxlen=window)
    return pd.DataFrame([json.loads(x) for x in lines])


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, out_html: Path) -> bool:
    try:
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            snapshot = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
            snapshot.save_html(str(out_html))
        except ImportError:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            r = Report(metrics=[DataDriftPreset()])
            r.run(reference_data=ref, current_data=cur)
            r.save_html(str(out_html))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Không tạo được báo cáo Evidently: {e}")
        return False


def run(cfg, current_texts=None, current_labels=None, source: str = "api_logs") -> dict:
    dc = cfg.drift
    train = load_split(cfg, "train_base")
    ref_sample = train.sample(min(dc.reference_samples, len(train)), random_state=cfg.seed)
    # Từ vựng xây trên phần train KHÔNG chứa mẫu tham chiếu -> tránh thiên lệch OOV.
    vocab = build_vocab(train.drop(ref_sample.index)["text"])

    if current_texts is None:
        logs = read_recent_logs(cfg_path(cfg, "logs_dir") / "requests.jsonl", dc.window)
        current_texts = logs["text"].tolist() if len(logs) else []
        current_labels = logs["label"].tolist() if len(logs) else []
    if len(current_texts) < 100:
        print(f"Chỉ có {len(current_texts)} mẫu hiện tại (<100), bỏ qua kiểm tra drift")
        return {"drift_detected": False, "skipped": True, "n_current": len(current_texts)}

    ref_feats = text_features(ref_sample["text"], vocab)
    cur_feats = text_features(current_texts, vocab)
    report = detect(ref_feats, cur_feats, dc.p_value, dc.share_threshold)
    report.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "source": source,
            "n_reference": len(ref_feats),
            "n_current": len(cur_feats),
        }
    )
    if current_labels:
        report["prediction_shift"] = label_shift(ref_sample["label"], current_labels)

    reports = cfg_path(cfg, "reports_dir")
    (reports / "drift").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    html = reports / "drift" / f"drift_{stamp}.html"
    report["evidently_html"] = str(html) if evidently_report(ref_feats, cur_feats, html) else None
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (reports / "drift" / f"drift_{stamp}.json").write_text(text)
    (reports / "drift_latest.json").write_text(text)
    print(
        f"Drift share={report['drift_share']:.2f} detected={report['drift_detected']} "
        + " ".join(f"{k}:{v['current_mean']:.3f}/{v['reference_mean']:.3f}" for k, v in report["features"].items())
    )
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-split", default=None, help="So sánh với một split thay vì log API")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.overrides)
    if args.current_split:
        df = load_split(cfg, args.current_split)
        return run(cfg, df["text"].tolist(), None, source=args.current_split)
    return run(cfg)


if __name__ == "__main__":
    main()
