"""Consent model: normalize session-scoped consent_state JSONB and household defaults."""
from __future__ import annotations

from typing import Any


def get_pipeline_settings():
    try:
        from config.settings import get_pipeline_settings as _get
        return _get()
    except Exception:
        return None


def normalize_consent_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalize sessions.consent_state into canonical keys with defaults.
    Keys: consent_share_with_caregiver, consent_share_text, consent_allow_outbound_contact,
          caregiver_contact_policy (jsonb: allowed_channels, quiet_hours, escalation_threshold).
    """
    raw = raw or {}
    settings = get_pipeline_settings()
    default_share = getattr(settings, "default_consent_share", True)
    default_watchlist = getattr(settings, "default_consent_watchlist", True)
    default_outbound = getattr(settings, "default_consent_allow_outbound", False)
    share_key = getattr(settings, "consent_share_key", "share_with_caregiver")
    watchlist_key = getattr(settings, "consent_watchlist_key", "watchlist_ok")
    outbound_key = getattr(settings, "consent_allow_outbound_key", "consent_allow_outbound_contact")

    # Support both canonical key and legacy allow_outbound_contact (e.g. from household_consent_defaults)
    outbound_val = raw.get(outbound_key)
    if outbound_val is None:
        outbound_val = raw.get("allow_outbound_contact", default_outbound)
    return {
        "consent_share_with_caregiver": raw.get(share_key, default_share),
        "consent_share_text": raw.get(share_key, default_share),
        "consent_allow_outbound_contact": outbound_val,
        "consent_watchlist_ok": raw.get(watchlist_key, default_watchlist),
        "caregiver_contact_policy": raw.get("caregiver_contact_policy") or {
            "allowed_channels": ["sms", "email"],
            "quiet_hours": {"start": "22:00", "end": "08:00"},
            "escalation_threshold": 4,
        },
    }
