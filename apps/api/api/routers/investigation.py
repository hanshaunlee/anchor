"""Investigation API: single user-facing workflow (financial + narrative + outreach candidates). Caregiver/admin only."""
from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from api.deps import get_supabase, require_user
from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/investigation", tags=["investigation"])


class InvestigationRunRequest(BaseModel):
    time_window_days: int = Field(7, ge=1, le=90, description="Events in last N days")
    dry_run: bool = Field(False, description="If true, no DB write; return preview")
    use_demo_events: bool = Field(False, description="If true, use built-in demo events")
    enqueue: bool = Field(
        False,
        description="If true, add job to processing_queue and return job_id; Modal or worker runs investigation in background.",
    )


@router.post("/run")
def run_investigation(
    body: InvestigationRunRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Run Investigation (financial detection + narrative enrichment + outreach candidates).
    Caregiver/admin only. When enqueue=true, job is added to the queue and Modal (or worker)
    runs it in background; returns job_id so UI can show "Running in background…" and refresh.
    """
    role = get_user_role(supabase, user_id)
    if role not in ("caregiver", "admin"):
        raise HTTPException(status_code=403, detail="Only caregivers or admins can run investigation")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    body = body or InvestigationRunRequest()

    if body.enqueue and not body.dry_run and not body.use_demo_events:
        try:
            window_bucket = date.today().isoformat()
            dedupe_key = f"{hh_id}|run_supervisor_ingest|{window_bucket}"
            row = {
                "household_id": hh_id,
                "job_type": "run_supervisor_ingest",
                "payload": {"time_window_days": body.time_window_days},
                "payload_version": 1,
                "status": "pending",
                "dedupe_key": dedupe_key,
            }
            ins = supabase.table("processing_queue").insert(row).execute()
            job_id = ins.data[0]["id"] if ins.data and len(ins.data) > 0 else None
            return {
                "ok": True,
                "enqueued": True,
                "job_id": job_id,
                "message": "Investigation queued; Modal or worker will run it in background. Refresh the page in a minute.",
                "supervisor_run_id": None,
                "mode": "enqueued",
                "child_run_ids": {},
                "created_signal_ids": [],
                "updated_signal_ids": [],
                "created_watchlist_ids": [],
                "outreach_candidates": [],
                "summary_json": {},
                "step_trace": [],
                "warnings": [],
            }
        except Exception as e:
            err = str(e).lower()
            if "unique" in err or "duplicate" in err or "dedupe" in err:
                return {
                    "ok": True,
                    "enqueued": False,
                    "message": "An investigation for this household is already queued.",
                    "supervisor_run_id": None,
                    "mode": "skipped",
                    "child_run_ids": {},
                    "created_signal_ids": [],
                    "updated_signal_ids": [],
                    "created_watchlist_ids": [],
                    "outreach_candidates": [],
                    "summary_json": {},
                    "step_trace": [],
                    "warnings": [],
                }
            raise HTTPException(status_code=500, detail=f"Enqueue failed: {e}")

    ingested_events = None
    if body.use_demo_events:
        from domain.agents.financial_security_agent import get_demo_events
        ingested_events = get_demo_events()
    result = run_supervisor(
        household_id=hh_id,
        supabase=supabase if not body.dry_run else None,
        run_mode=INGEST_PIPELINE,
        dry_run=body.dry_run,
        time_window_days=body.time_window_days,
        ingested_events=ingested_events,
    )
    return {
        "ok": True,
        "enqueued": False,
        "supervisor_run_id": result.get("supervisor_run_id"),
        "mode": result.get("mode"),
        "child_run_ids": result.get("child_run_ids"),
        "created_signal_ids": result.get("created_signal_ids"),
        "updated_signal_ids": result.get("updated_signal_ids"),
        "created_watchlist_ids": result.get("created_watchlist_ids"),
        "outreach_candidates": result.get("outreach_candidates"),
        "summary_json": result.get("summary_json"),
        "step_trace": result.get("step_trace"),
        "warnings": result.get("warnings"),
    }
