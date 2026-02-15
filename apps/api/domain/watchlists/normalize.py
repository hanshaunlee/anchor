"""Normalize watchlist values and compute fingerprint for deduplication."""
from __future__ import annotations

import hashlib
import re


def _normalize_phone(value: str) -> tuple[str, str]:
    """Strip non-digits, preserve leading +/country; display pretty."""
    digits = re.sub(r"\D", "", value)
    if not digits:
        return "", value
    normalized = digits
    if value.strip().startswith("+"):
        normalized = "+" + digits
    if len(digits) >= 10:
        display = f"+1 ({digits[-10:-7]}) {digits[-7:-4]}-{digits[-4:]}" if len(digits) == 10 or (len(digits) == 11 and digits[0] == "1") else f"+{digits}"
    else:
        display = normalized
    return normalized, display


def _normalize_email(value: str) -> tuple[str, str]:
    """Lowercase email."""
    v = (value or "").strip().lower()
    return v, v


def _normalize_phrase_or_topic(value: str) -> tuple[str, str]:
    """Lower, collapse whitespace."""
    v = (value or "").strip().lower()
    v = " ".join(v.split())
    return v, v


def normalize_watchlist_value(
    category: str,
    type_: str,
    value: str | list | None,
) -> tuple[str, str]:
    """
    Return (value_normalized, display_value).
    Phone: strip non-digits, display pretty. Emails: lower. Phrases/topics: lower, collapse whitespace.
    value may be a list (e.g. keywords) in which case it is joined to a string.
    """
    if isinstance(value, list):
        value = ", ".join(str(x) for x in value) if value else ""
    raw = (value or "").strip() if isinstance(value, str) else ""
    if not raw:
        return "", ""
    cat = (category or "other").lower()
    typ = (type_ or "").lower()
    if cat == "contact" or typ in ("phone", "new_contact"):
        if "@" in raw:
            return _normalize_email(raw)
        return _normalize_phone(raw)
    if cat in ("phrase", "topic") or typ in ("risky_phrase", "risky_topic", "keyword"):
        return _normalize_phrase_or_topic(raw)
    if cat in ("device_policy", "bank") or typ in ("high_risk_mode", "bank_freeze_keyword"):
        return _normalize_phrase_or_topic(raw)
    return _normalize_phrase_or_topic(raw)


def watchlist_fingerprint(category: str, type_: str, key: str, value_normalized: str) -> str:
    """SHA256(category|type|key|value_normalized) for unique constraint."""
    payload = f"{category}|{type_}|{key}|{value_normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
