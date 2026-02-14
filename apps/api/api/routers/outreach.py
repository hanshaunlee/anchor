"""Outbound caregiver outreach: trigger and list outbound_actions. Caregiver/admin only for trigger; elder sees elder_safe only."""
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from supabase import Client

from api.deps import get_supabase, require_user, require_caregiver_or_admin
from api.schemas import OutboundActionResponse, OutreachActionRequest
from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/actions", tags=["outreach"])


def _redact_for_elder(row: dict) -> dict:
    """Return only elder-safe fields for elder role."""
    payload = row.get("payload") or {}
    return {
        "id": row.get("id"),
        "household_id": row.get("household_id"),
        "triggered_by_risk_signal_id": row.get("triggered_by_risk_signal_id"),
        "action_type": row.get("action_type"),
        "channel": row.get("channel"),
        "recipient_name": None,
        "recipient_contact": None,
        "payload": {"elder_safe_message": payload.get("elder_safe_message"), "status": row.get("status")},
        "status": row.get("status"),
        "provider": row.get("provider"),
        "provider_message_id": None,
        "error": None,
        "created_at": row.get("created_at"),
        "sent_at": row.get("sent_at"),
        "delivered_at": row.get("delivered_at"),
    }


def _row_to_response(row: dict, redact_contact: bool = False) -> dict:
    if redact_contact:
        row = _redact_for_elder(row)
    def _dt(v: Any) -> datetime | None:
        if v is None:
            return None
        s = str(v).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    out = {
        "id": row["id"],
        "household_id": row["household_id"],
        "triggered_by_risk_signal_id": row.get("triggered_by_risk_signal_id"),
        "action_type": row.get("action_type"),
        "channel": row.get("channel"),
        "recipient_name": row.get("recipient_name"),
        "recipient_contact": row.get("recipient_contact"),
        "recipient_contact_last4": row.get("recipient_contact_last4"),
        "payload": row.get("payload") or {},
        "status": row.get("status"),
        "provider": row.get("provider"),
        "provider_message_id": row.get("provider_message_id"),
        "error": row.get("error"),
        "created_at": _dt(row.get("created_at")),
        "sent_at": _dt(row.get("sent_at")),
        "delivered_at": _dt(row.get("delivered_at")),
    }
    return out


@router.post("/outreach")
def post_outreach(
    body: OutreachActionRequest,
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """
    Trigger caregiver outreach for a risk signal. Caregiver/admin only.
    Returns outbound_action, agent_run_id, preview (when dry_run).
    """
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    role = get_user_role(supabase, user_id) or "caregiver"
    consent_state = {}
    r = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", hh_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if r.data and len(r.data) > 0:
        consent_state = r.data[0].get("consent_state") or {}
    result = run_caregiver_outreach_agent(
        hh_id,
        supabase,
        risk_signal_id=str(body.risk_signal_id),
        dry_run=body.dry_run,
        consent_state=consent_state,
        user_role=role,
    )
    outbound_action = None
    if result.get("outbound_action_id") and not body.dry_run:
        row = (
            supabase.table("outbound_actions")
            .select("*")
            .eq("id", result["outbound_action_id"])
            .single()
            .execute()
        )
        if row.data:
            outbound_action = _row_to_response(row.data, redact_contact=False)
    preview = None
    if body.dry_run:
        preview = {
            "caregiver_message": (result.get("summary_json") or {}).get("channel"),
            "step_trace": result.get("step_trace"),
        }
    return {
        "ok": True,
        "outbound_action": outbound_action,
        "agent_run_id": result.get("run_id"),
        "preview": preview,
        "suppressed": (result.get("summary_json") or {}).get("suppressed", False),
        "sent": (result.get("summary_json") or {}).get("sent", False),
    }


@router.get("/outreach/{action_id}")
def get_outreach(
    action_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get one outbound action. Caregiver/admin see full; elder sees elder_safe_message only."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    role = get_user_role(supabase, user_id) or "elder"
    r = (
        supabase.table("outbound_actions")
        .select("*")
        .eq("id", str(action_id))
        .eq("household_id", hh_id)
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="Outbound action not found.")
    redact = role == "elder"
    return _row_to_response(r.data, redact_contact=redact)


@router.get("/outreach")
def list_outreach(
    household_id: UUID | None = Query(None, description="Filter by household (must match yours)"),
    limit: int = Query(20, ge=1, le=100),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """List recent outbound actions for the current user's household."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    if household_id and str(household_id) != hh_id:
        raise HTTPException(status_code=403, detail="household_id must match your household.")
    role = get_user_role(supabase, user_id) or "elder"
    r = (
        supabase.table("outbound_actions")
        .select("*")
        .eq("household_id", hh_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    redact = role == "elder"
    return {"actions": [_row_to_response(row, redact_contact=redact) for row in (r.data or [])]}


@router.get("/outreach/summary")
def get_outreach_summary(
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """Counts (sent, suppressed, failed) and recent outbound actions for Agents page. Caregiver/admin only."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded.")
    all_r = (
        supabase.table("outbound_actions")
        .select("status")
        .eq("household_id", hh_id)
        .execute()
    )
    counts = {"sent": 0, "suppressed": 0, "failed": 0, "queued": 0, "delivered": 0}
    for row in (all_r.data or []):
        s = (row.get("status") or "").lower()
        if s in counts:
            counts[s] += 1
    recent_r = (
        supabase.table("outbound_actions")
        .select("id, status, created_at, sent_at, error, triggered_by_risk_signal_id, channel, recipient_contact_last4")
        .eq("household_id", hh_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    data = recent_r.data or []
    return {
        "counts": counts,
        "recent": [
            {
                "id": x.get("id"),
                "status": x.get("status"),
                "created_at": x.get("created_at"),
                "sent_at": x.get("sent_at"),
                "error": x.get("error"),
                "triggered_by_risk_signal_id": x.get("triggered_by_risk_signal_id"),
                "channel": x.get("channel"),
                "recipient_contact_last4": x.get("recipient_contact_last4"),
            }
            for x in data
        ],
    }
