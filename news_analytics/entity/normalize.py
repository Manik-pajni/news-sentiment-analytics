from __future__ import annotations

import re

NOISE_PREFIXES = (
    "THE ",
    "CEO OF ",
    "CFO OF ",
    "PRESIDENT OF ",
    "GOVERNMENT OF ",
)

GENERIC_STOP = {
    "UNITED STATES",
    "UNITED KINGDOM",
    "GOVERNMENT",
    "POLICE",
    "MILITARY",
    "PRESIDENT",
    "CONGRESS",
    "SENATE",
    "COURT",
    "MEDIA",
    "REUTERS",
    "ASSOCIATED PRESS",
    "BLOOMBERG",
    "CHINA",
    "RUSSIA",
    "EUROPE",
    "NATO",
    "UNITED NATIONS",
    "WHITE HOUSE",
}


def clean_mention(text: str) -> str:
    text = (text or "").upper()
    text = re.sub(r"[^A-Z0-9\s&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for prefix in NOISE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def is_generic(mention_norm: str) -> bool:
    if not mention_norm or len(mention_norm) < 2:
        return True
    if mention_norm in GENERIC_STOP:
        return True
    tokens = mention_norm.split()
    if len(tokens) == 1 and mention_norm in GENERIC_STOP:
        return True
    return False
