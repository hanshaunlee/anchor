"""
Recurring Contacts Agent: identifies contacts/numbers that appear across multiple sessions or
repeatedly in the same session (returning callers, bursty contact attempts).
Contributes watchlist candidates as an additional weight—not the sole determinant—so the
watchlist reflects both risk scores and recurrence.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run_ctx, step
from domain.graph_service import normalize_events

logger = logging.getLogger(__name__)

AGENT_NAME = "recurring_contacts"

# Minimum distinct sessions or mention count to consider "recurring"
MIN_SESSIONS_RECURRING = 2
MIN_MENTIONS_RECURRING = 2
# Cap watchlist items from this agent so it doesn't dominate
MAX_RECURRING_WATCHLIST_ITEMS = 25
# Score weight: min(1.0, session_count / 5) so more sessions = higher weight
RECURRENCE_SCORE_DIVISOR = 5.0


def _ingest_events(household_id: str, time_window_days: int, supabase: Any) -> list[dict]:
    """Fetch events for household in time window (same pattern as financial agent)."""
    if not supabase:
        return []
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=time_window_days)).isoformat()
        r = (
            supabase.table("events")
            .select("id, session_id, device_id, ts, seq, event_type, payload")
            .gte("ts", since)
            .order("ts")
            .limit(5000)
            .execute()
        )
        if not r.data:
            return []
        # Filter to sessions that belong to household
        session_ids = list({str(e.get("session_id", "")) for e in r.data if e.get("session_id")})
        if not session_ids:
            return list(r.data or [])
        sessions_r = supabase.table("sessions").select("id").eq("household_id", household_id).in_("id", session_ids).execute()
        allowed = {str(s["id"]) for s in (sessions_r.data or [])}
        return [e for e in (r.data or []) if str(e.get("session_id", "")) in allowed]
    except Exception as e:
        logger.debug("Recurring agent ingest failed: %s", e)
        return []


def run_recurring_contacts_agent(
    household_id: str,
    supabase: Any,
    *,
    events: list[dict] | None = None,
    time_window_days: int = 7,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Compute recurring contact / returning-caller signals and return watchlist candidates.
    Uses events if provided (e.g. from supervisor); otherwise fetches from DB.
    Returns watchlist_items compatible with supervisor merge (same shape as financial/ring).
    """
    now = datetime.now(timezone.utc)
    ctx = AgentContext(household_id=household_id, supabase=supabase, dry_run=dry_run)
    step_trace: list[dict] = []
    started_at = now.isoformat()
    summary_json: dict[str, Any] = {"headline": "Recurring contacts", "watchlist_count": 0}

    # Step 1 — Get events
    with step(ctx, step_trace, "ingest"):
        if events is not None:
            evs = events
            step_trace[-1]["notes"] = "events_passed_in"
        else:
            evs = _ingest_events(household_id, time_window_days, supabase)
            step_trace[-1]["notes"] = f"fetched_{len(evs)}"
        step_trace[-1]["outputs_count"] = len(evs)

    if not evs:
        step_trace[-1]["status"] = "skip"
        run_id = persist_agent_run_ctx(ctx, AGENT_NAME, "completed", step_trace, summary_json)
        return {"step_trace": step_trace, "summary_json": summary_json, "run_id": run_id, "watchlist_items": []}

    # Step 2 — Normalize to get entities and mentions (session_id, entity_id per mention)
    with step(ctx, step_trace, "normalize"):
        try:
            utterances, entities, mentions, relationships = normalize_events(household_id, evs)
        except Exception as e:
            logger.warning("Recurring agent normalize failed: %s", e)
            step_trace[-1]["status"] = "error"
            step_trace[-1]["error"] = str(e)
            run_id = persist_agent_run_ctx(ctx, AGENT_NAME, "failed", step_trace, summary_json)
            return {"step_trace": step_trace, "summary_json": summary_json, "run_id": run_id, "watchlist_items": []}
        step_trace[-1]["outputs_count"] = len(mentions)
        step_trace[-1]["notes"] = f"{len(entities)} entities, {len(mentions)} mentions"

    # Step 3 — Count per entity: distinct sessions and total mentions
    entity_sessions: dict[str, set[str]] = defaultdict(set)
    entity_mention_count: dict[str, int] = defaultdict(int)
    for m in mentions:
        eid = m.get("entity_id")
        sid = m.get("session_id")
        if eid:
            entity_mention_count[eid] += 1
            if sid:
                entity_sessions[eid].add(str(sid))

    # Build entity_id -> canonical for display (from entities list)
    entity_canonical: dict[str, str] = {}
    entity_type_map: dict[str, str] = {}
    for e in entities:
        eid = e.get("id")
        if eid:
            entity_canonical[eid] = (e.get("canonical") or "").strip() or eid
            entity_type_map[eid] = e.get("entity_type", "entity")

    # Step 4 — Produce watchlist items for recurring contacts (phone, email, person preferred; others included)
    with step(ctx, step_trace, "synthesize_watchlist"):
        expires = (now + timedelta(days=7)).isoformat()
        watchlist_items: list[dict] = []
        # Sort by (session_count, mention_count) descending to prioritize strong recurrence
        candidates = []
        for eid, sessions in entity_sessions.items():
            n_sessions = len(sessions)
            n_mentions = entity_mention_count.get(eid, 0)
            if n_sessions >= MIN_SESSIONS_RECURRING or n_mentions >= MIN_MENTIONS_RECURRING:
                etype = entity_type_map.get(eid, "entity")
                # Prefer phone, email, person for "contact" watchlist
                if etype not in ("phone", "email", "person"):
                    etype = "entity"
                candidates.append((eid, n_sessions, n_mentions, etype))
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for eid, n_sessions, n_mentions, etype in candidates[:MAX_RECURRING_WATCHLIST_ITEMS]:
            score = min(1.0, (n_sessions + n_mentions * 0.2) / RECURRENCE_SCORE_DIVISOR)
            canonical = entity_canonical.get(eid, eid)
            reason = f"Recurring contact (appeared in {n_sessions} session(s), {n_mentions} mention(s))"
            pattern = {
                "entity_id": eid,
                "entity_type": etype,
                "canonical": canonical[:200] if canonical else None,
                "score": score,
                "session_count": n_sessions,
                "mention_count": n_mentions,
            }
            watchlist_items.append({
                "watch_type": "recurring_contact",
                "pattern": pattern,
                "reason": reason,
                "priority": 3,  # Below high-risk (1) and risky_topic (2), but visible
                "expires_at": expires,
            })
        step_trace[-1]["outputs_count"] = len(watchlist_items)
        summary_json["watchlist_count"] = len(watchlist_items)
        summary_json["headline"] = f"Recurring contacts: {len(watchlist_items)} candidate(s)"

    run_id = persist_agent_run_ctx(ctx, AGENT_NAME, "completed", step_trace, summary_json)
    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "run_id": run_id,
        "watchlist_items": watchlist_items,
    }
