"""Agents API: Financial Security Agent and status/trace."""
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client

from api.broadcast import broadcast_risk_signal
from api.deps import get_supabase, require_user
from api.agents.financial_agent import run_financial_security_playbook

router = APIRouter(prefix="/agents", tags=["agents"])


def _household_id(supabase: Client, user_id: str) -> str | None:
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    return u.data["household_id"] if u.data else None


class FinancialRunRequest(BaseModel):
    household_id: UUID | None = Field(None, description="Override household (must belong to user); default from auth")
    time_window_days: int = Field(7, ge=1, le=90, description="Events in last N days")
    dry_run: bool = Field(False, description="If true, do not write to DB; return payload for preview")


class AgentStatusItem(BaseModel):
    agent_name: str
    last_run_at: datetime | None
    last_run_status: str | None
    last_run_summary: dict | None


class AgentsStatusResponse(BaseModel):
    agents: list[AgentStatusItem]


@router.post("/financial/run")
def run_financial_agent(
    body: FinancialRunRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Run the Financial Security Agent playbook for the household.
    Body: optional household_id (infer from auth), time_window_days (default 7), dry_run (default false).
    If dry_run=true: no DB write; returns computed risk_signals and watchlists for preview.
    """
    body = body or FinancialRunRequest()
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    if body.household_id is not None and str(body.household_id) != hh_id:
        raise HTTPException(status_code=403, detail="household_id must match your household")
    household_id = hh_id
    # Consent from latest session for this household (or default)
    consent_state = {}
    sess = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", household_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if sess.data and len(sess.data) > 0:
        consent_state = sess.data[0].get("consent_state") or {}
    result = run_financial_security_playbook(
        household_id=household_id,
        time_window_days=body.time_window_days,
        consent_state=consent_state,
        ingested_events=None,
        supabase=supabase if not body.dry_run else None,
        dry_run=body.dry_run,
    )
    # Broadcast each new risk signal to /ws/risk_signals
    for payload in result.get("inserted_signals_for_broadcast", []):
        broadcast_risk_signal(payload)
    return {
        "ok": True,
        "dry_run": body.dry_run,
        "run_id": result.get("run_id"),
        "risk_signals_count": len(result.get("risk_signals", [])),
        "watchlists_count": len(result.get("watchlists", [])),
        "inserted_signal_ids": result.get("inserted_signal_ids", []),
        "logs": result.get("logs", []),
        "risk_signals": result.get("risk_signals", []) if body.dry_run else None,
        "watchlists": result.get("watchlists", []) if body.dry_run else None,
    }


@router.get("/status", response_model=AgentsStatusResponse)
def get_agents_status(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """What agents exist and last run time / status / summary."""
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        return AgentsStatusResponse(agents=[])
    r = (
        supabase.table("agent_runs")
        .select("agent_name, started_at, ended_at, status, summary_json")
        .eq("household_id", hh_id)
        .order("started_at", desc=True)
        .execute()
    )
    rows = r.data or []
    # One row per agent_name (latest run)
    by_agent: dict[str, dict] = {}
    for row in rows:
        name = row.get("agent_name", "")
        if name not in by_agent:
            by_agent[name] = row
    agents_list = []
    for name, row in by_agent.items():
        started = row.get("started_at")
        agents_list.append(AgentStatusItem(
            agent_name=name,
            last_run_at=datetime.fromisoformat(started.replace("Z", "+00:00")) if started else None,
            last_run_status=row.get("status"),
            last_run_summary=row.get("summary_json"),
        ))
    # Ensure financial_security is listed even if never run
    if "financial_security" not in by_agent:
        agents_list.append(AgentStatusItem(
            agent_name="financial_security",
            last_run_at=None,
            last_run_status=None,
            last_run_summary=None,
        ))
    return AgentsStatusResponse(agents=agents_list)


@router.get("/financial/trace")
def get_financial_trace(
    run_id: UUID = Query(..., description="Agent run id from POST /agents/financial/run"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get trace for a financial agent run (from agent_runs)."""
    hh_id = _household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("agent_runs")
        .select("*")
        .eq("id", str(run_id))
        .eq("household_id", hh_id)
        .eq("agent_name", "financial_security")
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="Run not found")
    row = r.data
    return {
        "id": row["id"],
        "household_id": row["household_id"],
        "agent_name": row["agent_name"],
        "started_at": row["started_at"],
        "ended_at": row.get("ended_at"),
        "status": row.get("status"),
        "summary_json": row.get("summary_json"),
    }
