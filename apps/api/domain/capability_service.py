"""Household capabilities: explicit registry for notify, device push, bank connector and controls."""
from __future__ import annotations

from typing import Any

from supabase import Client

# Columns that exist in migration 013 (run_migrations_011_012_013). Migration 015 adds auto_send_outreach.
# We only persist these so the app works before 015 is run; auto_send_outreach is returned as False until then.
PERSISTED_CAPABILITY_KEYS = (
    "notify_sms_enabled",
    "notify_email_enabled",
    "device_policy_push_enabled",
    "bank_data_connector",
    "bank_control_capabilities",
)

DEFAULT_CAPABILITIES = {
    "notify_sms_enabled": False,
    "notify_email_enabled": False,
    "device_policy_push_enabled": True,
    "bank_data_connector": "none",
    "bank_control_capabilities": {
        "lock_card": False,
        "disable_transfers": False,
        "enable_alerts": True,
        "open_dispute": False,
    },
    "auto_send_outreach": False,
}


def get_household_capabilities(supabase: Client, household_id: str) -> dict[str, Any]:
    """Get capabilities for household; insert default row if missing."""
    r = (
        supabase.table("household_capabilities")
        .select(",".join(["household_id", "updated_at"] + list(PERSISTED_CAPABILITY_KEYS)))
        .eq("household_id", household_id)
        .limit(1)
        .execute()
    )
    if r.data and len(r.data) > 0:
        row = r.data[0]
        return {
            "household_id": row["household_id"],
            "notify_sms_enabled": row.get("notify_sms_enabled", False),
            "notify_email_enabled": row.get("notify_email_enabled", False),
            "device_policy_push_enabled": row.get("device_policy_push_enabled", True),
            "bank_data_connector": row.get("bank_data_connector", "none"),
            "bank_control_capabilities": row.get("bank_control_capabilities") or DEFAULT_CAPABILITIES["bank_control_capabilities"],
            "auto_send_outreach": row.get("auto_send_outreach", False),
            "updated_at": row.get("updated_at"),
        }
    # Upsert default (only columns that exist in schema 013)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    payload = {"household_id": household_id, "updated_at": now}
    for k in PERSISTED_CAPABILITY_KEYS:
        payload[k] = DEFAULT_CAPABILITIES[k]
    supabase.table("household_capabilities").upsert(payload, on_conflict="household_id").execute()
    return {
        "household_id": household_id,
        **DEFAULT_CAPABILITIES,
        "updated_at": now,
    }


def update_household_capabilities(
    supabase: Client,
    household_id: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Patch capabilities (caregiver/admin). Only allowed keys. Only persist columns that exist in DB (013)."""
    allowed = {
        "notify_sms_enabled", "notify_email_enabled", "device_policy_push_enabled",
        "bank_data_connector", "bank_control_capabilities", "auto_send_outreach",
    }
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    payload = {k: v for k, v in patch.items() if k in allowed}
    if not payload:
        return get_household_capabilities(supabase, household_id)
    payload["updated_at"] = now
    # Only send columns that exist in household_capabilities (013); skip auto_send_outreach until migration 015
    persist_payload = {"household_id": household_id, "updated_at": now}
    for k in PERSISTED_CAPABILITY_KEYS:
        if k in payload:
            persist_payload[k] = payload[k]
    supabase.table("household_capabilities").upsert(persist_payload, on_conflict="household_id").execute()
    return get_household_capabilities(supabase, household_id)
