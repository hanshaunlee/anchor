"""
Agent framework: shared context, step tracing, persist, and artifact helpers.
All agents use AgentContext and step() for consistent step_trace and DB persistence.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class AgentContext:
    """Shared context for agent runs: household, supabase, settings, dry_run, model_meta."""

    def __init__(
        self,
        household_id: str,
        supabase: Any | None,
        *,
        dry_run: bool = False,
        now: datetime | None = None,
        model_meta: dict[str, Any] | None = None,
        consent_state: dict[str, Any] | None = None,
    ):
        self.household_id = household_id
        self.supabase = supabase
        self.dry_run = dry_run
        self.now = now or datetime.now(timezone.utc)
        self.model_meta = model_meta or {}
        self.consent_state = consent_state or {}

    def consent_for(self, key: str, default: bool = True) -> bool:
        return bool(self.consent_state.get(key, default))


def _step_trace_entry(
    step: str,
    status: str,
    started_at: str,
    ended_at: str | None = None,
    inputs_count: int | None = None,
    outputs_count: int | None = None,
    notes: str | None = None,
    artifacts_refs: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    e: dict[str, Any] = {
        "step": step,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at or datetime.now(timezone.utc).isoformat(),
    }
    if inputs_count is not None:
        e["inputs_count"] = inputs_count
    if outputs_count is not None:
        e["outputs_count"] = outputs_count
    if notes:
        e["notes"] = notes
    if artifacts_refs:
        e["artifacts_refs"] = artifacts_refs
    if error:
        e["error"] = error
    return e


@contextmanager
def step(ctx: AgentContext, step_trace: list[dict], step_name: str):
    """Context manager that appends a step_trace entry with started_at/ended_at and status."""
    started = datetime.now(timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "step": step_name,
        "status": "ok",
        "started_at": started,
        "ended_at": None,
        "inputs_count": None,
        "outputs_count": None,
        "notes": None,
        "artifacts_refs": None,
    }
    step_trace.append(entry)
    try:
        yield entry
        entry["status"] = "ok"
    except Exception as e:
        logger.exception("Agent step %s failed: %s", step_name, e)
        entry["status"] = "error"
        entry["error"] = str(e)
        raise
    finally:
        entry["ended_at"] = datetime.now(timezone.utc).isoformat()


def persist_agent_run(
    supabase: Any,
    household_id: str,
    agent_name: str,
    *,
    started_at: str,
    ended_at: str,
    status: str,
    step_trace: list[dict],
    summary_json: dict[str, Any],
    dry_run: bool = False,
) -> str | None:
    """Insert agent_runs row. Returns run_id if persisted; None if dry_run or failure."""
    if dry_run:
        return None
    run_id = None
    try:
        ins = supabase.table("agent_runs").insert({
            "household_id": household_id,
            "agent_name": agent_name,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": status,
            "step_trace": step_trace,
            "summary_json": summary_json or {},
        }).execute()
        if ins.data and len(ins.data) > 0:
            run_id = ins.data[0].get("id")
    except Exception as e:
        logger.exception("persist_agent_run failed: %s", e)
    return run_id


def upsert_risk_signal(
    supabase: Any,
    household_id: str,
    payload: dict[str, Any],
    dry_run: bool,
) -> str | None:
    """Insert one risk_signal. Returns risk_signal id if persisted."""
    if dry_run:
        return None
    try:
        row = supabase.table("risk_signals").insert({
            "household_id": household_id,
            "signal_type": payload.get("signal_type", "risk_signal"),
            "severity": payload.get("severity", 2),
            "score": payload.get("score", 0.0),
            "explanation": payload.get("explanation", {}),
            "recommended_action": payload.get("recommended_action", {}),
            "status": payload.get("status", "open"),
        }).execute()
        if row.data and len(row.data) > 0:
            return row.data[0].get("id")
    except Exception as e:
        logger.exception("upsert_risk_signal failed: %s", e)
    return None


def upsert_watchlist(
    supabase: Any,
    household_id: str,
    payload: dict[str, Any],
    dry_run: bool,
) -> str | None:
    """Insert one watchlist row. Returns watchlist id if persisted."""
    if dry_run:
        return None
    try:
        row = supabase.table("watchlists").insert({
            "household_id": household_id,
            "watch_type": payload.get("watch_type", "entity_pattern"),
            "pattern": payload.get("pattern", {}),
            "reason": payload.get("reason"),
            "priority": payload.get("priority", 0),
            "expires_at": payload.get("expires_at"),
        }).execute()
        if row.data and len(row.data) > 0:
            return row.data[0].get("id")
    except Exception as e:
        logger.exception("upsert_watchlist failed: %s", e)
    return None


def upsert_summary(
    supabase: Any,
    household_id: str,
    *,
    period_start: str | None = None,
    period_end: str | None = None,
    summary_text: str,
    summary_json: dict[str, Any] | None = None,
    session_id: str | None = None,
    dry_run: bool = False,
) -> str | None:
    """Insert one summary row. Returns summary id if persisted."""
    if dry_run:
        return None
    try:
        row = supabase.table("summaries").insert({
            "household_id": household_id,
            "session_id": session_id,
            "period_start": period_start,
            "period_end": period_end,
            "summary_text": summary_text,
            "summary_json": summary_json or {},
        }).execute()
        if row.data and len(row.data) > 0:
            return row.data[0].get("id")
    except Exception as e:
        logger.exception("upsert_summary failed: %s", e)
    return None
