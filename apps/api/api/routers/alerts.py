"""Alerts API: alias for risk_signals with POST /alerts/{id}/refresh (supervisor NEW_ALERT)."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from domain.agents.supervisor import run_supervisor, NEW_ALERT
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/{id}/refresh")
def refresh_alert(
    id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Re-run supervisor NEW_ALERT for this alert (risk_signal). Caregiver/admin only. Same as POST /risk_signals/{id}/refresh."""
    role = get_user_role(supabase, user_id)
    if role not in ("caregiver", "admin"):
        raise HTTPException(status_code=403, detail="Only caregivers or admins can refresh alert")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    result = run_supervisor(
        household_id=hh_id,
        supabase=supabase,
        run_mode=NEW_ALERT,
        dry_run=False,
        risk_signal_id=str(id),
    )
    return {
        "ok": True,
        "supervisor_run_id": result.get("supervisor_run_id"),
        "mode": result.get("mode"),
        "child_run_ids": result.get("child_run_ids"),
        "summary_json": result.get("summary_json"),
        "step_trace": result.get("step_trace"),
    }
