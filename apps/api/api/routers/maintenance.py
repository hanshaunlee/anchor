"""System maintenance API: nightly model health. Admin or service role only."""
from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from domain.agents.supervisor import run_supervisor, NIGHTLY_MAINTENANCE
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/system/maintenance", tags=["system"])


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
