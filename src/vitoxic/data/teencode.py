"""Chèn nhiễu teencode / bỏ dấu để mô phỏng trôi dữ liệu ngôn ngữ mạng."""

from __future__ import annotations

import random
import re
import unicodedata

WORD_MAP: dict[str, list[str]] = {
    "không": ["ko", "k", "hông", "khum"],
    "được": ["dc", "đc", "đk"],
    "vậy": ["z", "v", "zậy"],
    "biết": ["bít", "bik"],
    "gì": ["j", "gì zậy"],
    "anh": ["a"],
    "em": ["e"],
    "yêu": ["iu"],
    "quá": ["wá", "qá"],
    "rồi": ["r", "rùi"],
    "thôi": ["thui", "thoy"],
    "với": ["vs", "zới"],
    "người": ["ng", "ngừi"],
    "mình": ["mk", "mik"],
    "bạn": ["bn", "b"],
    "thế": ["thía"],
    "chứ": ["chớ"],
    "luôn": ["lun"],
    "nhé": ["nha", "nhoa"],
    "trời": ["tr"],
    "chồng": ["ck"],
    "vợ": ["vk"],
    "cũng": ["cx", "cũm"],
    "thích": ["thik"],
    "đúng": ["đúg"],
    "trước": ["trc"],
    "chưa": ["chx"],
    "muốn": ["mún"],
    "đi": ["đii"],
    "ơi": ["ơiii"],
    "lắm": ["lém"],
    "nhiều": ["nhìu"],
    "vui": ["zui"],
    "và": ["vs", "zà"],
}

TEENCODE_TOKENS: frozenset[str] = frozenset(
    tok for variants in WORD_MAP.values() for v in variants for tok in v.split()
) | frozenset({"vl", "vcl", "dm", "đm", "cc", "ok", "oke", "kkk", "hihi", "huhu"})

_CHAR_RULES = [("ph", "f"), ("qu", "w"), ("gi", "j"), ("c", "k"), ("đ", "d"), ("y", "i")]
_TOKEN_RE = re.compile(r"(\w+|\W+)", re.UNICODE)


def remove_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text.replace("đ", "d").replace("Đ", "D"))


def _apply_char_rule(word: str, rng: random.Random) -> str:
    applicable = [(a, b) for a, b in _CHAR_RULES if a in word]
    if not applicable:
        return remove_diacritics(word)
    a, b = rng.choice(applicable)
    return word.replace(a, b)


def teencodify(text: str, prob: float, rng: random.Random | None = None) -> str:
    """Biến đổi từng từ với xác suất `prob`."""
    if prob <= 0:
        return text
    rng = rng or random.Random()
    out = []
    for tok in _TOKEN_RE.findall(text):
        if not tok.strip() or not tok[0].isalnum() or rng.random() >= prob:
            out.append(tok)
            continue
        low = tok.lower()
        if low in WORD_MAP:
            out.append(rng.choice(WORD_MAP[low]))
        elif rng.random() < 0.5:
            out.append(_apply_char_rule(low, rng))
        else:
            out.append(remove_diacritics(tok))
    # Kéo dài ký tự cuối câu kiểu "quáaaa" cho một số bình luận.
    result = "".join(out)
    if result and rng.random() < prob * 0.3:
        result = result + result[-1] * rng.randint(2, 4)
    return result
