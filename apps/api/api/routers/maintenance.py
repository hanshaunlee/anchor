"""System maintenance API: nightly model health, clear risk signals. Admin or service role only."""
from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from domain.agents.supervisor import run_supervisor, NIGHTLY_MAINTENANCE
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/system/maintenance", tags=["system"])


@router.post("/clear_risk_signals")
def clear_risk_signals(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Delete all risk signals for the current user's household. Admin or caregiver only.
    Use to start alerts from scratch. Cascades to risk_signal_embeddings and feedback.
    """
    role = get_user_role(supabase, user_id)
    if role not in ("admin", "caregiver"):
        raise HTTPException(status_code=403, detail="Only admin or caregiver can clear risk signals")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="No household")
    r = supabase.table("risk_signals").delete().eq("household_id", hh_id).execute()
    count = len(r.data) if r.data else 0
    return {"ok": True, "household_id": hh_id, "deleted_count": count}


@router.post("/run")
def run_maintenance(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Run NIGHTLY_MAINTENANCE (model_health agent only). Admin only.
    In production, call from cron; API enforces admin role.
    """
    role = get_user_role(supabase, user_id)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Only admin can run system maintenance")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    result = run_supervisor(
        household_id=hh_id,
        supabase=supabase,
        run_mode=NIGHTLY_MAINTENANCE,
        dry_run=False,
        env="prod",
    )
    return {
        "ok": True,
        "supervisor_run_id": result.get("supervisor_run_id"),
        "mode": result.get("mode"),
        "child_run_ids": result.get("child_run_ids"),
        "summary_json": result.get("summary_json"),
        "step_trace": result.get("step_trace"),
    }
