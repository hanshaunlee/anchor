"""Graph domain: normalize events to utterances, entities, mentions, relationships. Single place for build and optional persist (RLS)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ml.graph.builder import GraphBuilder

logger = logging.getLogger(__name__)


def build_graph_from_events(
    household_id: str,
    events: list[dict[str, Any]],
    *,
    supabase: Any | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Build graph structures from raw events using GraphBuilder (single orchestration).
    Optionally persists entities and relationships to Supabase when supabase is provided (RLS applies).
    Returns (utterances, entities, mentions, relationships).
    """
    utterances, entities, mentions, relationships = normalize_events(household_id, events)
    if supabase and household_id:
        _persist_graph_artifacts(supabase, household_id, entities, relationships)
    return utterances, entities, mentions, relationships


def _persist_graph_artifacts(
    supabase: Any,
    household_id: str,
    entities: list[dict],
    relationships: list[dict],
) -> None:
    """Upsert entities then relationships; map builder string ids to DB UUIDs. RLS applies."""
    id_map: dict[str, str] = {}
    for e in entities:
        eid = e.get("id")
        if not eid:
            continue
        etype = e.get("entity_type", "entity")
        canonical = (e.get("canonical") or "").strip() or eid
        canonical_hash = e.get("canonical_hash")
        try:
            q = (
                supabase.table("entities")
                .select("id")
                .eq("household_id", household_id)
                .eq("entity_type", etype)
            )
            if canonical_hash:
                q = q.eq("canonical_hash", canonical_hash)
            else:
                q = q.eq("canonical", canonical).is_("canonical_hash", "NULL")
            r = q.limit(1).execute()
            if r.data and len(r.data) > 0:
                id_map[eid] = str(r.data[0]["id"])
            else:
                ins = (
                    supabase.table("entities")
                    .insert({
                        "household_id": household_id,
                        "entity_type": etype,
                        "canonical": canonical,
                        "canonical_hash": canonical_hash or None,
                    })
                    .execute()
                )
                if ins.data and len(ins.data) > 0:
                    id_map[eid] = str(ins.data[0]["id"])
        except Exception as ex:
            logger.debug("Entity persist failed for %s: %s", eid, ex)

    for rel in relationships:
        src = id_map.get(str(rel.get("src_entity_id", "")))
        dst = id_map.get(str(rel.get("dst_entity_id", "")))
        if not src or not dst:
            continue
        first_ts = rel.get("first_seen_at")
        last_ts = rel.get("last_seen_at")
        if isinstance(first_ts, (int, float)):
            first_ts = datetime.fromtimestamp(float(first_ts), tz=timezone.utc).isoformat()
        if isinstance(last_ts, (int, float)):
            last_ts = datetime.fromtimestamp(float(last_ts), tz=timezone.utc).isoformat()
        if not first_ts or not last_ts:
            continue
        try:
            supabase.table("relationships").upsert(
                {
                    "household_id": household_id,
                    "src_entity_id": src,
                    "dst_entity_id": dst,
                    "rel_type": rel.get("rel_type", "CO_OCCURS"),
                    "weight": float(rel.get("weight", 1.0)),
                    "first_seen_at": first_ts,
                    "last_seen_at": last_ts,
                },
                on_conflict="household_id,src_entity_id,dst_entity_id,rel_type",
            ).execute()
        except Exception as ex:
            logger.debug("Relationship upsert failed: %s", ex)


def normalize_events(
    household_id: str,
    events: list[dict[str, Any]],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Build graph structures from raw events using GraphBuilder.
    Returns (utterances, entities, mentions, relationships). No DB write.
    """
    builder = GraphBuilder(household_id)
    by_session: dict[str, list] = {}
    for ev in events:
        sid = ev.get("session_id") or ""
        if isinstance(sid, dict):
            sid = sid.get("id", "")
        sid = str(sid)
        if sid not in by_session:
            by_session[sid] = []
        by_session[sid].append(ev)
    for sid, evs in by_session.items():
        evs_sorted = sorted(evs, key=lambda e: (e.get("ts") or "", e.get("seq", 0)))
        device_id = evs_sorted[0].get("device_id", "") if evs_sorted else ""
        if isinstance(device_id, dict):
            device_id = device_id.get("id", "")
        builder.process_events(evs_sorted, sid, str(device_id))
    return (
        builder.get_utterance_list(),
        builder.get_entity_list(),
        builder.get_mention_list(),
        builder.get_relationship_list(),
    )
