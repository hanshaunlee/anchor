"""Agents API: Financial Security Agent, Graph Drift, Evidence Narrative, Ring Discovery, Calibration, Red-Team; status/trace."""
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client

from api.broadcast import broadcast_risk_signal
from api.deps import get_supabase, require_user
from domain.agents.financial_security_agent import get_demo_events, run_financial_security_playbook
from domain.agents.graph_drift_agent import run_graph_drift_agent
from domain.agents.evidence_narrative_agent import run_evidence_narrative_agent
from domain.agents.ring_discovery_agent import run_ring_discovery_agent
from domain.agents.continual_calibration_agent import run_continual_calibration_agent
from domain.agents.synthetic_redteam_agent import run_synthetic_redteam_agent
from domain.agents.registry import get_known_agent_names, slug_to_agent_name
from domain.ingest_service import get_household_id

router = APIRouter(prefix="/agents", tags=["agents"])

KNOWN_AGENTS = get_known_agent_names()


class FinancialRunRequest(BaseModel):
    household_id: UUID | None = Field(None, description="Override household (must belong to user); default from auth")
    time_window_days: int = Field(7, ge=1, le=90, description="Events in last N days")
    dry_run: bool = Field(False, description="If true, do not write to DB; return payload for preview")
    use_demo_events: bool = Field(False, description="If true, use built-in demo events (synthetic scam scenario) instead of DB; response includes input_events")


class AgentRunRequest(BaseModel):
    dry_run: bool = Field(True, description="If true, no DB write; return preview (step_trace, summary)")


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
    Body: optional household_id, time_window_days (default 7), dry_run (default false), use_demo_events (default false).
    If dry_run=true: no DB write; returns computed risk_signals and watchlists for preview.
    If use_demo_events=true: run on built-in demo events (synthetic scam scenario); response includes input_events.
    """
    body = body or FinancialRunRequest()
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    if body.household_id is not None and str(body.household_id) != hh_id:
        raise HTTPException(status_code=403, detail="household_id must match your household")

    consent_state = {}
    sess = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", hh_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if sess.data and len(sess.data) > 0:
        consent_state = sess.data[0].get("consent_state") or {}

    ingested_events = get_demo_events() if body.use_demo_events else None
    result = run_financial_security_playbook(
        household_id=hh_id,
        time_window_days=body.time_window_days,
        consent_state=consent_state,
        ingested_events=ingested_events,
        supabase=supabase if not body.dry_run else None,
        dry_run=body.dry_run,
    )

    for payload in result.get("inserted_signals_for_broadcast", []):
        broadcast_risk_signal(payload)

    out = {
        "ok": True,
        "dry_run": body.dry_run,
        "use_demo_events": body.use_demo_events,
        "run_id": result.get("run_id"),
        "risk_signals_count": len(result.get("risk_signals", [])),
        "watchlists_count": len(result.get("watchlists", [])),
        "inserted_signal_ids": result.get("inserted_signal_ids", []),
        "logs": result.get("logs", []),
        "motif_tags": result.get("motif_tags", []),
        "timeline_snippet": result.get("timeline_snippet", []),
        "risk_signals": result.get("risk_signals", []) if (body.dry_run or body.use_demo_events) else None,
        "watchlists": result.get("watchlists", []) if (body.dry_run or body.use_demo_events) else None,
    }
    if body.use_demo_events:
        out["input_events"] = get_demo_events()
    return out


@router.get("/financial/demo")
def run_financial_agent_demo():
    """
    Run the Financial Security Agent on built-in demo events (no auth).
    Returns full input (input_events) and output (logs, motif_tags, risk_signals, watchlists) for inspection.
    Does not write to DB. Use POST /agents/financial/run with use_demo_events=true for authenticated runs.
    """
    events = get_demo_events()
    result = run_financial_security_playbook(
        household_id="demo",
        time_window_days=7,
        consent_state={"share_with_caregiver": True, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )
    return {
        "ok": True,
        "message": "Demo run (no DB write, no auth). Use POST /agents/financial/run with use_demo_events=true for authenticated runs.",
        "input_events": events,
        "input_summary": f"{len(events)} events: Medicare urgency + share_ssn intent + phone 555-1234",
        "output": {
            "logs": result.get("logs", []),
            "motif_tags": result.get("motif_tags", []),
            "timeline_snippet": result.get("timeline_snippet", []),
            "risk_signals": result.get("risk_signals", []),
            "watchlists": result.get("watchlists", []),
        },
        "risk_signals_count": len(result.get("risk_signals", [])),
        "watchlists_count": len(result.get("watchlists", [])),
    }


@router.get("/status", response_model=AgentsStatusResponse)
def get_agents_status(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """What agents exist and last run time / status / summary."""
    hh_id = get_household_id(supabase, user_id)
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
    by_agent: dict[str, dict] = {}
    for row in rows:
        name = row.get("agent_name", "")
        if name not in by_agent:
            by_agent[name] = row
    agents_list = []
    for name in KNOWN_AGENTS:
        row = by_agent.get(name)
        agents_list.append(AgentStatusItem(
            agent_name=name,
            last_run_at=datetime.fromisoformat(row["started_at"].replace("Z", "+00:00")) if row and row.get("started_at") else None,
            last_run_status=row.get("status") if row else None,
            last_run_summary=row.get("summary_json") if row else None,
        ))
    return AgentsStatusResponse(agents=agents_list)


@router.get("/trace")
def get_agent_trace(
    run_id: UUID = Query(..., description="Agent run id"),
    agent_name: str = Query(..., description="Agent name (e.g. financial_security, graph_drift)"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get trace for any agent run (step_trace + summary). Friendly for UI as 'Agent Trace'."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("agent_runs")
        .select("*")
        .eq("id", str(run_id))
        .eq("household_id", hh_id)
        .eq("agent_name", agent_name)
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="Run not found")
    row = r.data
    out = {
        "id": row["id"],
        "household_id": row["household_id"],
        "agent_name": row["agent_name"],
        "started_at": row["started_at"],
        "ended_at": row.get("ended_at"),
        "status": row.get("status"),
        "summary_json": row.get("summary_json"),
    }
    if "step_trace" in row:
        out["step_trace"] = row["step_trace"]
    return out


@router.get("/{agent_slug}/trace")
def get_agent_trace_by_slug(
    agent_slug: str,
    run_id: UUID = Query(..., description="Agent run id"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get trace for an agent run by slug (e.g. drift, narrative, ring). Resolves slug to agent_name."""
    agent_name = slug_to_agent_name(agent_slug)
    if agent_name is None:
        if agent_slug == "financial":
            agent_name = "financial_security"
        else:
            raise HTTPException(status_code=404, detail="Unknown agent slug")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("agent_runs")
        .select("*")
        .eq("id", str(run_id))
        .eq("household_id", hh_id)
        .eq("agent_name", agent_name)
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="Run not found")
    row = r.data
    out = {
        "id": row["id"],
        "household_id": row["household_id"],
        "agent_name": row["agent_name"],
        "started_at": row["started_at"],
        "ended_at": row.get("ended_at"),
        "status": row.get("status"),
        "summary_json": row.get("summary_json"),
    }
    if "step_trace" in row:
        out["step_trace"] = row["step_trace"]
    return out


def _persist_agent_run(supabase: Client, household_id: str, agent_name: str, result: dict) -> dict:
    """Insert agent_runs row, then update with step_trace/ended_at/status/summary_json. Returns run_id if persisted."""
    run_id = None
    try:
        ins = supabase.table("agent_runs").insert({
            "household_id": household_id,
            "agent_name": agent_name,
            "started_at": result.get("started_at"),
            "status": result.get("status", "completed"),
            "summary_json": result.get("summary_json") or {},
            "step_trace": result.get("step_trace") or [],
        }).execute()
        if ins.data and len(ins.data) > 0:
            run_id = ins.data[0].get("id")
            supabase.table("agent_runs").update({
                "ended_at": result.get("ended_at"),
                "status": result.get("status", "completed"),
                "summary_json": result.get("summary_json") or {},
                "step_trace": result.get("step_trace") or [],
            }).eq("id", run_id).execute()
    except Exception:
        pass
    return run_id


@router.post("/drift/run")
def run_drift_agent(
    body: AgentRunRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Graph Drift Agent: multi-metric embedding drift; drift_warning risk_signal if detected. Dry-run returns preview."""
    body = body or AgentRunRequest()
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    result = run_graph_drift_agent(hh_id, supabase=supabase, dry_run=body.dry_run)
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json")}


@router.post("/narrative/run")
def run_narrative_agent(
    body: AgentRunRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Evidence Narrative Agent: evidence-grounded narrative; redaction-aware. Dry-run returns preview."""
    body = body or AgentRunRequest()
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    result = run_evidence_narrative_agent(hh_id, supabase=supabase, dry_run=body.dry_run)
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json")}


@router.post("/ring/run")
def run_ring_agent(
    body: AgentRunRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Ring Discovery Agent: interaction graph clustering; ring_candidate risk_signals and rings table. Dry-run returns preview."""
    body = body or AgentRunRequest()
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    result = run_ring_discovery_agent(hh_id, supabase=supabase, neo4j_available=False, dry_run=body.dry_run)
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json")}


@router.post("/calibration/run")
def run_calibration_agent(
    body: AgentRunRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Continual Calibration Agent: Platt/conformal from feedback; calibration report. Dry-run returns preview."""
    body = body or AgentRunRequest()
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    result = run_continual_calibration_agent(hh_id, supabase=supabase, dry_run=body.dry_run)
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json")}


@router.post("/redteam/run")
def run_redteam_agent(
    body: AgentRunRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Synthetic Red-Team Agent: scenario DSL + regression harness; pass rate and failing_cases. Dry-run default."""
    body = body or AgentRunRequest()
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    result = run_synthetic_redteam_agent(hh_id, supabase=supabase, dry_run=body.dry_run)
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json")}


@router.get("/financial/trace")
def get_financial_trace(
    run_id: UUID = Query(..., description="Agent run id from POST /agents/financial/run"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get trace for a financial agent run (from agent_runs)."""
    hh_id = get_household_id(supabase, user_id)
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
    out = {
        "id": row["id"],
        "household_id": row["household_id"],
        "agent_name": row["agent_name"],
        "started_at": row["started_at"],
        "ended_at": row.get("ended_at"),
        "status": row.get("status"),
        "summary_json": row.get("summary_json"),
    }
    if "step_trace" in row:
        out["step_trace"] = row["step_trace"]
    return out
