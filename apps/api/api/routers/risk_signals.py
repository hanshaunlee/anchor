from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import (
    FeedbackSubmit,
    RiskSignalDetail,
    RiskSignalListResponse,
    RiskSignalStatus,
    SimilarIncidentsResponse,
)
from domain.explain_service import get_similar_incidents
from domain.ingest_service import get_household_id
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


@router.get("/{signal_id}", response_model=RiskSignalDetail)
def get_risk_signal_route(
    signal_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Full detail including explanation_json and evidence pointers (session_ids, event_ids, entity_ids)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    detail = get_risk_signal_detail(signal_id, hh_id, supabase)
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