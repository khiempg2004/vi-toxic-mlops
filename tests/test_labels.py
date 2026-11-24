import unicodedata

import pytest

from vitoxic.labels import LABELS, build_messages, parse_label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CLEAN", "CLEAN"),
        ("offensive", "OFFENSIVE"),
        ("HATE<|im_end|>", "HATE"),
        ("  Nhãn: HATE", "HATE"),
        ("Đây là OFFENSIVE, không phải HATE", "OFFENSIVE"),
        ("không biết", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_label(raw, expected):
    assert parse_label(raw) == expected


def test_build_messages_truncates_and_normalizes():
    msgs = build_messages("a" * 1000, max_chars=100)
    assert msgs[0]["role"] == "system"
    assert all(lab in msgs[0]["content"] for lab in LABELS)
    assert msgs[1]["content"] == "Bình luận: " + "a" * 100
    nfd = unicodedata.normalize("NFD", "không")
    assert build_messages(nfd)[1]["content"].endswith(unicodedata.normalize("NFC", "không"))
