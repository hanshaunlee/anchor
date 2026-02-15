"""Ring persist with fingerprint dedupe and canonical view."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from domain.rings.fingerprint import ring_fingerprint, jaccard_overlap

logger = logging.getLogger(__name__)

JACCARD_MERGE_THRESHOLD = 0.9


def upsert_ring_by_fingerprint(
    supabase: Any,
    household_id: str,
    member_entity_ids: list[str],
    score: float,
    meta: dict[str, Any] | None = None,
    summary_label: str | None = None,
    summary_text: str | None = None,
    signals_count: int = 0,
) -> str | None:
    """
    Insert or update a ring: match by fingerprint or Jaccard overlap >= threshold.
    Returns ring_id. Updates existing ring (members, score, meta, updated_at) when matched.
    """
    if not supabase or not household_id or not member_entity_ids:
        return None
    now = datetime.now(timezone.utc).isoformat()
    fp = ring_fingerprint(member_entity_ids)
    member_set = set(member_entity_ids)
    meta = meta or {}

    # Prefer exact fingerprint match
    try:
        existing = (
            supabase.table("rings")
            .select("id, fingerprint, meta")
            .eq("household_id", household_id)
            .eq("status", "active")
            .eq("fingerprint", fp)
            .limit(1)
            .execute()
        )
        if existing.data and len(existing.data) > 0:
            ring_id = existing.data[0]["id"]
            _update_ring(supabase, ring_id, member_entity_ids, score, meta, summary_label, summary_text, signals_count, now)
            return str(ring_id)
    except Exception as e:
        logger.debug("Rings fingerprint lookup failed (column may not exist): %s", e)

    # Soft match: fetch active rings and compare Jaccard
    try:
        all_active = (
            supabase.table("rings")
            .select("id, meta")
            .eq("household_id", household_id)
            .eq("status", "active")
            .execute()
        )
        for row in (all_active.data or []):
            rid = row.get("id")
            # Get current members for this ring
            mem_r = supabase.table("ring_members").select("entity_id").eq("ring_id", rid).execute()
            current = set(str(m.get("entity_id", "")) for m in (mem_r.data or []) if m.get("entity_id"))
            if jaccard_overlap(member_set, current) >= JACCARD_MERGE_THRESHOLD:
                _update_ring(supabase, rid, member_entity_ids, score, meta, summary_label, summary_text, signals_count, now)
                return str(rid)
    except Exception as e:
        logger.debug("Rings Jaccard lookup failed: %s", e)

    # Insert new ring (with optional columns when migration 021 applied)
    try:
        row: dict[str, Any] = {
            "household_id": household_id,
            "score": float(score),
            "meta": meta,
            "status": "active",
            "fingerprint": fp,
            "first_seen_at": now,
            "signals_count": signals_count,
        }
        if summary_label is not None:
            row["summary_label"] = summary_label
        if summary_text is not None:
            row["summary_text"] = summary_text
        ins = supabase.table("rings").insert(row).execute()
        if not ins.data or len(ins.data) == 0:
            return None
        ring_id = ins.data[0].get("id")
        if not ring_id:
            return None
        for eid in member_entity_ids[:50]:
            try:
                supabase.table("ring_members").insert({
                    "ring_id": ring_id,
                    "entity_id": eid,
                    "role": "member",
                    "first_seen_at": now,
                    "last_seen_at": now,
                }).execute()
            except Exception:
                pass
        return str(ring_id)
    except Exception as e:
        # Fallback when migration 021 not applied: minimal insert
        logger.debug("Ring insert with fingerprint failed, trying minimal: %s", e)
        try:
            row = {"household_id": household_id, "score": float(score), "meta": meta}
            ins = supabase.table("rings").insert(row).execute()
            if not ins.data or len(ins.data) == 0:
                return None
            ring_id = ins.data[0].get("id")
            if not ring_id:
                return None
            for eid in member_entity_ids[:50]:
                try:
                    supabase.table("ring_members").insert({
                        "ring_id": ring_id,
                        "entity_id": eid,
                        "role": "member",
                    }).execute()
                except Exception:
                    pass
            return str(ring_id)
        except Exception as e2:
            logger.warning("Ring insert failed: %s", e2)
            return None


def _update_ring(
    supabase: Any,
    ring_id: str,
    member_entity_ids: list[str],
    score: float,
    meta: dict[str, Any],
    summary_label: str | None,
    summary_text: str | None,
    signals_count: int,
    now: str,
) -> None:
    """Update existing ring row and replace members."""
    try:
        update_payload: dict[str, Any] = {
            "updated_at": now,
            "score": float(score),
            "meta": meta,
            "signals_count": signals_count,
        }
        if summary_label is not None:
            update_payload["summary_label"] = summary_label
        if summary_text is not None:
            update_payload["summary_text"] = summary_text
        supabase.table("rings").update(update_payload).eq("id", ring_id).execute()
        # Replace members: delete existing then insert (simple approach)
        supabase.table("ring_members").delete().eq("ring_id", ring_id).execute()
        for eid in member_entity_ids[:50]:
            try:
                supabase.table("ring_members").insert({
                    "ring_id": ring_id,
                    "entity_id": eid,
                    "role": "member",
                    "first_seen_at": now,
                    "last_seen_at": now,
                }).execute()
            except Exception:
                pass
    except Exception as e:
        logger.warning("Ring update failed for %s: %s", ring_id, e)
