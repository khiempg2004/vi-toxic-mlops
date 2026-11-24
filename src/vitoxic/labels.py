"""Nhãn, câu chỉ dẫn và hàm phân tích đầu ra của LLM.

Dùng chung cho huấn luyện, đánh giá và phục vụ để đảm bảo prompt nhất quán.
"""

from __future__ import annotations

import unicodedata

LABELS: list[str] = ["CLEAN", "OFFENSIVE", "HATE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

SYSTEM_PROMPT = (
    "Bạn là hệ thống kiểm duyệt bình luận mạng xã hội tiếng Việt. "
    "Hãy phân loại bình luận vào đúng một nhãn:\n"
    "- CLEAN: bình luận bình thường, không xúc phạm.\n"
    "- OFFENSIVE: có từ ngữ thô tục, khiếm nhã nhưng không nhắm vào cá nhân hay nhóm cụ thể.\n"
    "- HATE: công kích, lăng mạ, thù ghét nhắm vào cá nhân hoặc nhóm người.\n"
    "Chỉ trả lời đúng một từ: CLEAN, OFFENSIVE hoặc HATE."
)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def truncate(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    return text if len(text) <= max_chars else text[:max_chars]


def build_messages(text: str, max_chars: int = 512) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Bình luận: {truncate(text, max_chars)}"},
    ]


def parse_label(output: str | None) -> str | None:
    """Ánh xạ đầu ra tự do của LLM về một nhãn; None nếu không nhận ra."""
    if not output:
        return None
    up = output.strip().upper()
    for label in LABELS:
        if up.startswith(label):
            return label
    positions = {label: up.find(label) for label in LABELS if label in up}
    if positions:
        return min(positions, key=positions.__getitem__)
    return None
