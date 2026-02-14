from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import IngestEventsRequest, IngestEventsResponse
from domain.ingest_service import get_household_id, ingest_events

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/events", response_model=IngestEventsResponse)
def ingest_events_route(
    body: IngestEventsRequest,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Batch ingest event packets. Device authenticated; events must reference session/device in same household."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    try:
        return ingest_events(body, hh_id, supabase)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))