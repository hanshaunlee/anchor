from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import IngestEventsRequest, IngestEventsResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/events", response_model=IngestEventsResponse)
def ingest_events(
    body: IngestEventsRequest,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Batch ingest event packets. Device authenticated; events must reference session/device in same household."""
    if not body.events:
        return IngestEventsResponse(ingested=0, session_ids=[], last_ts=None)
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    if not u.data:
        raise HTTPException(status_code=403, detail="No household")
    hh_id = u.data["household_id"]
    # Verify all sessions belong to household
    session_ids = list({e.session_id for e in body.events})
    sessions = supabase.table("sessions").select("id, household_id").in_("id", [str(s) for s in session_ids]).execute()
    for s in sessions.data or []:
        if s["household_id"] != hh_id:
            raise HTTPException(status_code=403, detail="Session not in your household")
    rows = []
    last_ts = None
    for e in body.events:
        ts = e.ts.isoformat() if isinstance(e.ts, datetime) else e.ts
        last_ts = e.ts
        rows.append({
            "session_id": str(e.session_id),
            "device_id": str(e.device_id),
            "ts": ts,
            "seq": e.seq,
            "event_type": e.event_type,
            "payload": e.payload,
            "payload_version": e.payload_version,
        })
    supabase.table("events").insert(rows).execute()
    return IngestEventsResponse(ingested=len(rows), session_ids=session_ids, last_ts=last_ts)
