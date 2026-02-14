from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import EventListItem, EventsListResponse, SessionListItem, SessionListResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
def list_sessions(
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """List sessions for current household. UI: from/to for date range; consent_state for redaction."""
    # Get household_id
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    if not u.data:
        return SessionListResponse(sessions=[], total=0)
    hh_id = u.data["household_id"]

    q = supabase.table("sessions").select("id, device_id, started_at, ended_at, mode, consent_state", count="exact")
    q = q.eq("household_id", hh_id).order("started_at", desc=True).range(offset, offset + limit - 1)
    if from_ts:
        q = q.gte("started_at", from_ts.isoformat())
    if to_ts:
        q = q.lte("started_at", to_ts.isoformat())
    r = q.execute()

    sessions_data = r.data or []
    total = r.count or 0
    session_ids = [s["id"] for s in sessions_data]
    summaries = {}
    if session_ids:
        sum_r = supabase.table("summaries").select("session_id, summary_text").in_("session_id", session_ids).execute()
        for row in sum_r.data or []:
            if row.get("session_id"):
                summaries[str(row["session_id"])] = row.get("summary_text")

    sessions = [
        SessionListItem(
            id=UUID(s["id"]),
            device_id=UUID(s["device_id"]),
            started_at=datetime.fromisoformat(s["started_at"].replace("Z", "+00:00")),
            ended_at=datetime.fromisoformat(s["ended_at"].replace("Z", "+00:00")) if s.get("ended_at") else None,
            mode=s["mode"],
            consent_state=s.get("consent_state") or {},
            summary_text=summaries.get(s["id"]),
        )
        for s in sessions_data
    ]
    return SessionListResponse(sessions=sessions, total=total)


@router.get("/{session_id}/events", response_model=EventsListResponse)
def list_session_events(
    session_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Paginated events for a session. Redacted by consent; payload may omit text."""
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    if not u.data:
        return EventsListResponse(events=[], total=0, next_offset=None)
    # Verify session belongs to household
    s = supabase.table("sessions").select("id").eq("id", str(session_id)).eq("household_id", u.data["household_id"]).single().execute()
    if not s.data:
        return EventsListResponse(events=[], total=0, next_offset=None)

    q = (
        supabase.table("events")
        .select("id, ts, seq, event_type, payload, text_redacted", count="exact")
        .eq("session_id", str(session_id))
        .order("ts")
        .range(offset, offset + limit - 1)
    )
    r = q.execute()
    data = r.data or []
    total = r.count or 0
    next_offset = offset + len(data) if offset + len(data) < total else None
    events = [
        EventListItem(
            id=UUID(e["id"]),
            ts=datetime.fromisoformat(e["ts"].replace("Z", "+00:00")),
            seq=e["seq"],
            event_type=e["event_type"],
            payload=e.get("payload") or {},
            text_redacted=e.get("text_redacted", True),
        )
        for e in data
    ]
    return EventsListResponse(events=events, total=total, next_offset=next_offset)
