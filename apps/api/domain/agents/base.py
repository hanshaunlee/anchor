"""
Agent framework: shared context, step tracing, persist, and artifact helpers.
All agents use AgentContext and step() for consistent step_trace and DB persistence.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class AgentContext:
    """Shared context for agent runs: household, supabase, settings, dry_run, model_meta, consent, user_role."""

    def __init__(
        self,
        household_id: str,
        supabase: Any | None,
        *,
        dry_run: bool = False,
        now: datetime | None = None,
        model_meta: dict[str, Any] | None = None,
        consent_state: dict[str, Any] | None = None,
        user_role: str | None = None,
        settings: Any | None = None,
        ml_settings: Any | None = None,
    ):
        self.household_id = household_id
        self.supabase = supabase
        self.dry_run = dry_run
        self.now = now or datetime.now(timezone.utc)
        self.model_meta = model_meta or {}
        self.consent_state = consent_state or {}
        self.user_role = user_role or "caregiver"
        self._settings = settings
        self._ml_settings = ml_settings

    @property
    def settings(self) -> Any:
        if self._settings is not None:
            return self._settings
        try:
            from config.settings import get_pipeline_settings
            return get_pipeline_settings()
        except Exception:
            return None

    @property
    def ml_settings(self) -> Any:
        if self._ml_settings is not None:
            return self._ml_settings
        try:
            from config.settings import get_ml_settings
            return get_ml_settings()
        except Exception:
            return None

    def consent_for(self, key: str, default: bool = True) -> bool:
        return bool(self.consent_state.get(key, default))


@contextmanager
def step(
    ctx: AgentContext,
    step_trace: list[dict],
    step_name: str,
    *,
    inputs_count: int | None = None,
    notes: str | None = None,
):
    """Context manager that appends a step_trace entry with started_at/ended_at, status, optional inputs_count/notes."""
    started = datetime.now(timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "step": step_name,
        "status": "ok",
        "started_at": started,
        "ended_at": None,
        "inputs_count": inputs_count,
        "outputs_count": None,
        "notes": notes,
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
    artifacts_refs: dict[str, Any] | None = None,
) -> str | None:
    """Insert agent_runs row. Returns run_id if persisted; None if dry_run or failure."""
    if dry_run:
        return None
    summary = dict(summary_json or {})
    if artifacts_refs:
        summary["artifact_refs"] = artifacts_refs
    run_id = None
    try:
        ins = supabase.table("agent_runs").insert({
            "household_id": household_id,
            "agent_name": agent_name,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": status,
            "step_trace": step_trace,
            "summary_json": summary,
        }).execute()
        if ins.data and len(ins.data) > 0:
            run_id = ins.data[0].get("id")
    except Exception as e:
        logger.exception("persist_agent_run failed: %s", e)
    return run_id


def persist_agent_run_ctx(
    ctx: AgentContext,
    agent_name: str,
    status: str,
    step_trace: list[dict],
    summary_json: dict[str, Any],
    artifacts_refs: dict[str, Any] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> str | None:
    """Persist agent run using context. started_at/ended_at from first/last step if not provided."""
    if not ctx.supabase:
        return None
    start = started_at or (step_trace[0].get("started_at") if step_trace else datetime.now(timezone.utc).isoformat())
    end = ended_at or (step_trace[-1].get("ended_at") if step_trace else datetime.now(timezone.utc).isoformat())
    return persist_agent_run(
        ctx.supabase,
        ctx.household_id,
        agent_name,
        started_at=start,
        ended_at=end,
        status=status,
        step_trace=step_trace,
        summary_json=summary_json,
        dry_run=ctx.dry_run,
        artifacts_refs=artifacts_refs,
    )


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


def upsert_risk_signal_ctx(ctx: AgentContext, signal_type: str, severity: int, score: float, explanation_json: dict, recommended_action_json: dict, status: str = "open") -> str | None:
    """Insert one risk_signal using context."""
    if not ctx.supabase:
        return None
    return upsert_risk_signal(
        ctx.supabase,
        ctx.household_id,
        {"signal_type": signal_type, "severity": severity, "score": score, "explanation": explanation_json, "recommended_action": recommended_action_json, "status": status},
        ctx.dry_run,
    )


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


def upsert_watchlists(ctx: AgentContext, watchlist_items: list[dict[str, Any]]) -> list[str]:
    """Insert multiple watchlist rows. Returns list of inserted ids."""
    ids: list[str] = []
    if not ctx.supabase or ctx.dry_run:
        return ids
    for w in watchlist_items:
        wid = upsert_watchlist(ctx.supabase, ctx.household_id, w, False)
        if wid:
            ids.append(wid)
    return ids


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


def upsert_summary_ctx(
    ctx: AgentContext,
    summary_type: str,
    period_start: str | None,
    period_end: str | None,
    summary_text: str,
    summary_json: dict[str, Any] | None = None,
) -> str | None:
    """Insert one summary using context."""
    if not ctx.supabase:
        return None
    summary_json = dict(summary_json or {})
    summary_json["summary_type"] = summary_type
    return upsert_summary(
        ctx.supabase,
        ctx.household_id,
        period_start=period_start,
        period_end=period_end,
        summary_text=summary_text,
        summary_json=summary_json,
        dry_run=ctx.dry_run,
    )


def fetch_household_inputs(ctx: AgentContext, time_window_days: int = 7) -> dict[str, Any]:
    """
    Fetch events, sessions, existing signals count, embeddings availability, calibration for the household.
    Returns dict with keys: events, sessions, signals_count, embeddings_available, calibration, consent_state.
    """
    out: dict[str, Any] = {
        "events": [],
        "sessions": [],
        "signals_count": 0,
        "embeddings_available": False,
        "calibration": None,
        "consent_state": ctx.consent_state,
    }
    if not ctx.supabase:
        return out
    try:
        end = ctx.now
        start = end - timedelta(days=time_window_days)
        start_iso = start.isoformat()
        end_iso = end.isoformat()
        sess_r = (
            ctx.supabase.table("sessions")
            .select("id, started_at, consent_state")
            .eq("household_id", ctx.household_id)
            .gte("started_at", start_iso)
            .lte("started_at", end_iso)
            .execute()
        )
        out["sessions"] = sess_r.data or []
        session_ids = [s["id"] for s in out["sessions"]]
        if session_ids:
            for sid in session_ids[:50]:
                ev_r = ctx.supabase.table("events").select("id, session_id, device_id, ts, seq, event_type, payload").eq("session_id", sid).order("ts").limit(500).execute()
                out["events"].extend(ev_r.data or [])
        sig_r = ctx.supabase.table("risk_signals").select("id", count="exact").eq("household_id", ctx.household_id).gte("ts", start_iso).execute()
        out["signals_count"] = getattr(sig_r, "count", None) or len(sig_r.data or [])
        emb_r = (
            ctx.supabase.table("risk_signal_embeddings")
            .select("risk_signal_id")
            .eq("household_id", ctx.household_id)
            .eq("has_embedding", True)
            .gte("created_at", start_iso)
            .limit(1)
            .execute()
        )
        out["embeddings_available"] = bool(emb_r.data and len(emb_r.data) > 0)
        cal_r = ctx.supabase.table("household_calibration").select("*").eq("household_id", ctx.household_id).limit(1).execute()
        out["calibration"] = cal_r.data[0] if cal_r.data and len(cal_r.data) > 0 else None
    except Exception as e:
        logger.debug("fetch_household_inputs failed: %s", e)
    return out
