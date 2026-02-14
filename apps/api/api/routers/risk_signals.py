from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import (
    FeedbackSubmit,
    PlaybookDetail,
    RiskSignalDetail,
    RiskSignalListResponse,
    RiskSignalStatus,
    SimilarIncidentsResponse,
)
from domain.explain_service import get_similar_incidents, run_deep_dive_explainer
from domain.ingest_service import get_household_id, get_user_role
from domain.risk_service import get_risk_signal_detail, list_risk_signals, submit_feedback

router = APIRouter(prefix="/risk_signals", tags=["risk_signals"])


@router.get("", response_model=RiskSignalListResponse)
def list_risk_signals_route(
    status: RiskSignalStatus | None = Query(None),
    severity_min: int | None = Query(None, alias="severity>=", ge=1, le=5),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """List risk signals. UI: filter by status and minimum severity."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    return list_risk_signals(hh_id, supabase, status=status, severity_min=severity_min, limit=limit, offset=offset)


def _consent_allows_share_text(supabase: Client, household_id: str) -> bool:
    """True if latest session consent allows sharing text with caregiver."""
    r = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", household_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        return True
    consent = r.data[0].get("consent_state") or {}
    return consent.get("share_with_caregiver", True)


@router.get("/{signal_id}", response_model=RiskSignalDetail)
def get_risk_signal_route(
    signal_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Full detail including explanation_json and evidence pointers (session_ids, event_ids, entity_ids). Redacted when consent disallows sharing text."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    consent_ok = _consent_allows_share_text(supabase, hh_id)
    detail = get_risk_signal_detail(signal_id, hh_id, supabase, consent_allows_share_text=consent_ok)
    if not detail:
        raise HTTPException(status_code=404, detail="Risk signal not found")
    return detail


@router.get("/{signal_id}/similar", response_model=SimilarIncidentsResponse)
def get_similar_incidents_route(
    signal_id: UUID,
    top_k: int = Query(5, ge=1, le=20),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Similar Incidents: retrieve nearest neighbors by embedding (cosine). Shows 3–5 most similar past incidents and outcomes."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    return get_similar_incidents(signal_id, hh_id, supabase, top_k=top_k)


@router.post("/{signal_id}/explain/deep_dive")
def post_deep_dive_explain_route(
    signal_id: UUID,
    mode: str = Query("pg", description="Explainer mode: pg (PGExplainer) or gnn (GNNExplainer; not yet implemented)"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Run deep-dive explainer and persist deep_dive_subgraph on the risk signal. mode=pg copies model_subgraph; mode=gnn returns 501."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    if mode not in ("pg", "gnn"):
        raise HTTPException(status_code=400, detail="mode must be 'pg' or 'gnn'")
    try:
        result = run_deep_dive_explainer(signal_id, hh_id, supabase, mode=mode)
        return result
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))


@router.post("/{signal_id}/feedback")
def submit_feedback_route(
    signal_id: UUID,
    body: FeedbackSubmit,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Caregiver labels: true_positive / false_positive / unsure. Stored in feedback table."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    try:
        submit_feedback(signal_id, hh_id, body, user_id, supabase)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@router.get("/{signal_id}/playbook", response_model=PlaybookDetail)
def get_risk_signal_playbook(
    signal_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Shortcut: get playbook for this risk signal (latest active/completed)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    pb = (
        supabase.table("action_playbooks")
        .select("*")
        .eq("risk_signal_id", str(signal_id))
        .eq("household_id", hh_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not pb.data or len(pb.data) == 0:
        raise HTTPException(status_code=404, detail="No playbook for this signal")
    playbook_id = pb.data[0]["id"]
    role = get_user_role(supabase, user_id) or "elder"
    from api.routers.playbooks import _get_playbook_with_tasks
    data = _get_playbook_with_tasks(supabase, str(playbook_id), hh_id, role)
    if not data:
        raise HTTPException(status_code=404, detail="Playbook not found")
    return PlaybookDetail(
        id=UUID(data["id"]),
        household_id=UUID(data["household_id"]),
        risk_signal_id=UUID(data["risk_signal_id"]),
        playbook_type=data["playbook_type"],
        graph=data.get("graph") or {},
        status=data["status"],
        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
        tasks=data["tasks"],
    )