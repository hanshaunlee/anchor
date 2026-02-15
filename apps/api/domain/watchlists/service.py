"""Watchlist items: batch upsert, dedupe by fingerprint, sticky types."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from domain.watchlists.normalize import normalize_watchlist_value, watchlist_fingerprint

logger = logging.getLogger(__name__)

STICKY_TYPES = frozenset({"high_risk_mode", "device_policy", "bank_freeze_keyword"})


def _map_legacy_to_item(
    household_id: str,
    batch_id: str,
    raw: dict[str, Any],
    source_agent: str,
    source_run_id: str | None,
    evidence_signal_ids: list[str] | None,
) -> dict[str, Any]:
    """Map legacy watchlist payload (watch_type, pattern, reason) to watchlist_items row."""
    watch_type = raw.get("watch_type", "entity_pattern")
    pattern = raw.get("pattern") or {}
    reason = raw.get("reason") or ""
    priority = int(raw.get("priority", 5))
    expires_at = raw.get("expires_at")

    category = "other"
    type_ = watch_type
    key = "item"
    value = ""
    if isinstance(pattern, dict):
        if pattern.get("entity_type") in ("phone", "email", "person"):
            category = "contact"
            type_ = "new_contact"
            key = "contact_to_watch"
            value = pattern.get("canonical") or pattern.get("canonical_hash") or ""
        elif pattern.get("keywords") or pattern.get("topic_hash"):
            category = "topic"
            type_ = "risky_topic"
            key = "risky_topic"
            value = pattern.get("keywords") or pattern.get("topic_hash") or ""
        elif "high_risk" in str(watch_type).lower() or "device" in str(watch_type).lower():
            category = "device_policy"
            type_ = "high_risk_mode"
            key = "high_risk_mode"
            value = "enabled"
        elif pattern.get("entity_id"):
            category = "other"
            type_ = "entity_pattern"
            key = "entity_id"
            value = str(pattern.get("entity_id"))
        else:
            key = pattern.get("entity_type") or watch_type
            value = str(pattern.get("canonical") or pattern.get("canonical_hash") or "")

    value_normalized, display_value = normalize_watchlist_value(category, type_, value or reason)
    if not value_normalized and value:
        value_normalized = (value or "").strip().lower()[:500]
    if not display_value and value:
        display_value = (value or "")[:200]

    display_label = {"contact": "Contact", "phrase": "Phrase", "topic": "Topic", "device_policy": "Device protection", "bank": "Bank", "other": "Item"}.get(category, "Item")
    fingerprint = watchlist_fingerprint(category, type_, key, value_normalized)

    return {
        "household_id": household_id,
        "batch_id": batch_id,
        "status": "active",
        "category": category,
        "type": type_,
        "key": key,
        "value": (value or "")[:1000],
        "value_normalized": (value_normalized or "")[:500],
        "display_label": display_label,
        "display_value": (display_value or "")[:500],
        "explanation": (reason or "")[:2000],
        "priority": priority,
        "score": float(pattern.get("score")) if isinstance(pattern.get("score"), (int, float)) else None,
        "source_agent": source_agent,
        "source_run_id": source_run_id,
        "evidence_signal_ids": evidence_signal_ids or [],
        "expires_at": expires_at,
        "fingerprint": fingerprint,
    }


def upsert_watchlist_batch(
    supabase: Any,
    household_id: str,
    batch_id: str,
    items: list[dict[str, Any]],
    source_agent: str,
    source_run_id: str | None = None,
    evidence_signal_ids: list[str] | None = None,
) -> list[str]:
    """
    Normalize each item, upsert by fingerprint (update if active), mark previous batch items superseded.
    Sticky types (high_risk_mode, device_policy) are updated in place, not duplicated.
    Returns list of watchlist_item ids (inserted or updated).
    """
    if not supabase or not household_id or not items:
        return []
    batch_id_uuid = batch_id if isinstance(batch_id, str) else str(batch_id)
    evidence = evidence_signal_ids or []
    now = datetime.now(timezone.utc).isoformat()
    ids_out: list[str] = []

    normalized_rows = [
        _map_legacy_to_item(household_id, batch_id_uuid, w, source_agent, source_run_id, evidence)
        for w in items
    ]

    for row in normalized_rows:
        row["updated_at"] = now
        fingerprint = row["fingerprint"]
        try:
            existing = (
                supabase.table("watchlist_items")
                .select("id, status")
                .eq("household_id", household_id)
                .eq("fingerprint", fingerprint)
                .eq("status", "active")
                .limit(1)
                .execute()
            )
            if existing.data and len(existing.data) > 0:
                rec = existing.data[0]
                supabase.table("watchlist_items").update({
                    "updated_at": now,
                    "batch_id": batch_id_uuid,
                    "priority": row["priority"],
                    "score": row["score"],
                    "explanation": row["explanation"],
                    "evidence_signal_ids": row["evidence_signal_ids"],
                    "expires_at": row["expires_at"],
                }).eq("id", rec["id"]).execute()
                ids_out.append(rec["id"])
            else:
                ins = supabase.table("watchlist_items").insert(row).execute()
                if ins.data and len(ins.data) > 0:
                    ids_out.append(ins.data[0]["id"])
        except Exception as e:
            logger.exception("watchlist_items upsert failed for fingerprint %s: %s", fingerprint[:16], e)

    try:
        active_fingerprints_this_batch = {r["fingerprint"] for r in normalized_rows}
        sticky_fingerprints = {
            r["fingerprint"] for r in normalized_rows
            if r.get("type") in STICKY_TYPES
        }
        supersede = (
            supabase.table("watchlist_items")
            .select("id, fingerprint, type")
            .eq("household_id", household_id)
            .eq("status", "active")
            .neq("batch_id", batch_id_uuid)
            .execute()
        )
        for rec in (supersede.data or []):
            fp = rec.get("fingerprint")
            typ = rec.get("type")
            if fp in active_fingerprints_this_batch or (typ in STICKY_TYPES and fp in sticky_fingerprints):
                continue
            supabase.table("watchlist_items").update({"status": "superseded", "updated_at": now}).eq("id", rec["id"]).execute()
    except Exception as e:
        logger.warning("watchlist_items supersede step failed: %s", e)

    return ids_out


def list_active_watchlist_items(
    supabase: Any,
    household_id: str,
    *,
    category: str | None = None,
    type_: str | None = None,
    source_agent: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """List active watchlist_items for household, optional filters. Returns rows as dicts."""
    if not supabase or not household_id:
        return []
    try:
        q = (
            supabase.table("watchlist_items")
            .select("*")
            .eq("household_id", household_id)
            .eq("status", "active")
            .order("priority", desc=True)
            .order("updated_at", desc=True)
            .limit(limit)
        )
        if category:
            q = q.eq("category", category)
        if type_:
            q = q.eq("type", type_)
        if source_agent:
            q = q.eq("source_agent", source_agent)
        r = q.execute()
        return list(r.data or [])
    except Exception as e:
        logger.debug("list_active_watchlist_items failed (table may not exist): %s", e)
        return []
