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
from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent
from domain.agents.incident_response_agent import run_incident_response_agent
from domain.agents.registry import get_agents_catalog, get_known_agent_names, slug_to_agent_name
from domain.ingest_service import get_household_id, get_user_role

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
    last_run_id: str | None = None
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
            "step_trace": result.get("step_trace", []),
        },
        "risk_signals_count": len(result.get("risk_signals", [])),
        "watchlists_count": len(result.get("watchlists", [])),
    }


@router.get("/catalog")
def get_agents_catalog_route(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Agent registry + visibility for current user (role, consent, env, calibration, model)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        return {"catalog": [], "role": "elder"}
    role = get_user_role(supabase, user_id) or "elder"
    consent = {}
    r = supabase.table("sessions").select("consent_state").eq("household_id", hh_id).order("started_at", desc=True).limit(1).execute()
    if r.data and len(r.data) > 0:
        consent = r.data[0].get("consent_state") or {}
    calibration_present = False
    try:
        cal = supabase.table("household_calibration").select("calibration_params").eq("household_id", hh_id).limit(1).execute()
        if cal.data and len(cal.data) > 0 and cal.data[0].get("calibration_params"):
            calibration_present = True
    except Exception:
        pass
    model_available = False
    try:
        emb = supabase.table("risk_signal_embeddings").select("risk_signal_id").eq("household_id", hh_id).eq("has_embedding", True).limit(1).execute()
        model_available = bool(emb.data and len(emb.data) > 0)
    except Exception:
        pass
    env = "prod"
    catalog = get_agents_catalog(role=role, consent=consent, environment=env, calibration_present=calibration_present, model_available=model_available)
    return {"catalog": catalog, "role": role}


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
        .select("id, agent_name, started_at, ended_at, status, summary_json")
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
            last_run_id=str(row["id"]) if row and row.get("id") else None,
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
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json"), "artifacts_refs": result.get("artifacts_refs")}


@router.get("/narrative/report/{report_id}")
def get_narrative_report(
    report_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get a persisted Evidence Narrative report by id (RLS: household)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = supabase.table("narrative_reports").select("id, household_id, agent_run_id, risk_signal_ids, report_json, created_at").eq("id", str(report_id)).eq("household_id", hh_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Report not found")
    return r.data[0]


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
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json"), "artifacts_refs": result.get("artifacts_refs")}


@router.get("/calibration/report")
def get_calibration_report(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Latest Continual Calibration run summary (for View calibration report UI)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("agent_runs")
        .select("id, started_at, summary_json")
        .eq("household_id", hh_id)
        .eq("agent_name", "continual_calibration")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="No calibration report found")
    row = r.data[0]
    return {"run_id": row.get("id"), "started_at": row.get("started_at"), "summary_json": row.get("summary_json") or {}}


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
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json"), "artifacts_refs": result.get("artifacts_refs")}


@router.get("/redteam/report")
def get_redteam_report(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Latest Synthetic Red-Team run summary and replay payload (for View report / Open in replay)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("agent_runs")
        .select("id, started_at, summary_json")
        .eq("household_id", hh_id)
        .eq("agent_name", "synthetic_redteam")
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="No red-team report found")
    row = r.data[0]
    return {"run_id": row.get("id"), "started_at": row.get("started_at"), "summary_json": row.get("summary_json") or {}}


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
    return {"ok": True, "dry_run": body.dry_run, "run_id": result.get("run_id"), "step_trace": result.get("step_trace"), "summary_json": result.get("summary_json"), "artifacts_refs": result.get("artifacts_refs")}


class OutreachRunRequest(BaseModel):
    risk_signal_id: UUID = Field(..., description="Risk signal to escalate")
    dry_run: bool = Field(True, description="If true, preview only; no send")


class IncidentResponseRunRequest(BaseModel):
    risk_signal_id: UUID = Field(..., description="Risk signal for incident response")
    dry_run: bool = Field(False, description="If true, no DB write; return preview")


@router.post("/outreach/run")
def run_outreach_agent(
    body: OutreachRunRequest,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Caregiver Outreach Agent: outbound notify/call/email. Caregiver/admin only; elder gets 403."""
    from domain.ingest_service import get_user_role
    role = get_user_role(supabase, user_id)
    if role not in ("caregiver", "admin"):
        raise HTTPException(status_code=403, detail="Only caregivers or admins can run outreach")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    consent_state = {}
    sess = supabase.table("sessions").select("consent_state").eq("household_id", hh_id).order("started_at", desc=True).limit(1).execute()
    if sess.data and len(sess.data) > 0:
        consent_state = sess.data[0].get("consent_state") or {}
    result = run_caregiver_outreach_agent(
        hh_id, supabase, risk_signal_id=str(body.risk_signal_id), dry_run=body.dry_run, consent_state=consent_state, user_role=role or "caregiver"
    )
    return {
        "ok": True,
        "dry_run": body.dry_run,
        "run_id": result.get("run_id"),
        "outbound_action_id": result.get("outbound_action_id"),
        "step_trace": result.get("step_trace"),
        "summary_json": result.get("summary_json"),
        "suppressed": (result.get("summary_json") or {}).get("suppressed", False),
        "sent": (result.get("summary_json") or {}).get("sent", False),
    }


@router.post("/incident-response/run")
def run_incident_response(
    body: IncidentResponseRunRequest,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Incident Response Agent: capability-aware playbook, incident packet, tasks. Dry run supported."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    consent_state = {}
    sess = supabase.table("sessions").select("consent_state").eq("household_id", hh_id).order("started_at", desc=True).limit(1).execute()
    if sess.data and len(sess.data) > 0:
        consent_state = sess.data[0].get("consent_state") or {}
    try:
        result = run_incident_response_agent(
            hh_id, str(body.risk_signal_id), supabase=supabase, dry_run=body.dry_run, consent_state=consent_state,
        )
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise
    return {
        "ok": True,
        "dry_run": body.dry_run,
        "run_id": result.get("run_id"),
        "playbook_id": result.get("playbook_id"),
        "incident_packet_id": result.get("incident_packet_id"),
        "step_trace": result.get("step_trace"),
        "summary_json": result.get("summary_json"),
    }


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
