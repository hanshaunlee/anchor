"""System API: run ingest pipeline (enqueue or run). Caregiver/admin only; server-side permission checks apply."""
from datetime import date
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client

from api.deps import get_supabase, require_user
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/system", tags=["system"])

PAYLOAD_VERSION_INGEST = 1


class RunIngestPipelineRequest(BaseModel):
    """Optional body for POST /system/run_ingest_pipeline."""
    time_window_days: int = Field(7, ge=1, le=90)
    enqueue: bool = Field(False, description="If true, insert into processing_queue; worker will run. If false, run inline (blocking).")
    dedupe: bool = Field(True, description="If true (default), use dedupe_key to avoid duplicate pending jobs for same household/day.")


@router.post("/run_ingest_pipeline")
def run_ingest_pipeline(
    body: RunIngestPipelineRequest | None = None,
    enqueue: bool = Query(False, description="Enqueue to processing_queue instead of running inline"),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """
    Run supervisor INGEST_PIPELINE for the current user's household.
    Caregiver/admin only (server-side checks). When enqueue=true, inserts into processing_queue
    with optional dedupe_key to prevent storms. When enqueue=false, runs inline (blocking).
    """
    role = get_user_role(supabase, user_id)
    if role not in ("caregiver", "admin"):
        raise HTTPException(status_code=403, detail="Only caregivers or admins can run ingest pipeline")
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")

    body = body or RunIngestPipelineRequest()
    do_enqueue = body.enqueue or enqueue
    time_window_days = body.time_window_days

    if do_enqueue:
        try:
            window_bucket = date.today().isoformat()
            dedupe_key = f"{hh_id}|run_supervisor_ingest|{window_bucket}" if body.dedupe else None
            row = {
                "household_id": hh_id,
                "job_type": "run_supervisor_ingest",
                "payload": {"time_window_days": time_window_days},
                "payload_version": PAYLOAD_VERSION_INGEST,
                "status": "pending",
            }
            if dedupe_key is not None:
                row["dedupe_key"] = dedupe_key
            ins = supabase.table("processing_queue").insert(row).execute()
            job_id = ins.data[0]["id"] if ins.data and len(ins.data) > 0 else None
            return {
                "ok": True,
                "enqueued": True,
                "job_id": job_id,
                "message": "Job enqueued; worker will run supervisor INGEST_PIPELINE.",
            }
        except Exception as e:
            err = str(e).lower()
            if "dedupe" in err or "unique" in err or "duplicate" in err:
                return {"ok": True, "enqueued": False, "message": "A job for this household is already pending (deduped)."}
            raise HTTPException(status_code=500, detail=f"Enqueue failed: {e}")

    # Inline run: supervisor INGEST_PIPELINE
    from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE
    result = run_supervisor(
        household_id=hh_id,
        supabase=supabase,
        run_mode=INGEST_PIPELINE,
        dry_run=False,
        time_window_days=time_window_days,
    )
    return {
        "ok": True,
        "enqueued": False,
        "supervisor_run_id": result.get("supervisor_run_id"),
        "mode": result.get("mode"),
        "created_signal_ids": result.get("created_signal_ids", []),
        "step_trace": result.get("step_trace", []),
    }
