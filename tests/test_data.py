import random

import pandas as pd
import pytest

from vitoxic.data.prepare import apply_teencode, normalize_columns, split_stream
from vitoxic.data.teencode import remove_diacritics, teencodify


def test_normalize_columns_vihsd_format():
    raw = pd.DataFrame({"free_text": ["  xin chào ", "vãi", "thằng ngu", None, ""], "label_id": [0, 1, 2, 0, 1]})
    out = normalize_columns(raw)
    assert list(out.columns) == ["text", "label"]
    assert out["label"].tolist() == ["CLEAN", "OFFENSIVE", "HATE"]
    assert out["text"].iloc[0] == "xin chào"


def test_normalize_columns_string_labels():
    out = normalize_columns(pd.DataFrame({"text": ["a", "b"], "label": ["clean", "HATE"]}))
    assert out["label"].tolist() == ["CLEAN", "HATE"]


def test_normalize_columns_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame({"foo": [1]}))


def test_split_stream_sizes(synth_df):
    base, batches = split_stream(synth_df, 0.7, 3, seed=1)
    assert len(base) + sum(len(b) for b in batches) == len(synth_df)
    assert len(batches) == 3
    assert set(base["label"]) == {"CLEAN", "OFFENSIVE", "HATE"}


def test_teencode_deterministic_and_changes_text():
    text = "anh không biết em được yêu quá rồi"
    a = teencodify(text, 1.0, random.Random(0))
    b = teencodify(text, 1.0, random.Random(0))
    assert a == b and a != text
    assert teencodify(text, 0.0, random.Random(0)) == text


def test_remove_diacritics():
    assert remove_diacritics("Đường không được") == "Duong khong duoc"


def test_apply_teencode_keeps_original(synth_df):
    out = apply_teencode(synth_df, 0.8, seed=0)
    assert (out["text_orig"] == synth_df["text"]).all()
    assert (out["text"] != synth_df["text"]).mean() > 0.5
