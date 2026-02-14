"""Event ingest: validate household/sessions and persist event batch.
Idempotent on (session_id, seq): re-sending the same event batch uses upsert so duplicates do not error."""
from datetime import datetime, timezone

from supabase import Client

from api.schemas import IngestEventsRequest, IngestEventsResponse


def get_household_id(supabase: Client, user_id: str) -> str | None:
    """Resolve user_id to household_id. Returns None if user has no row in public.users (e.g. not onboarded)."""
    r = supabase.table("users").select("household_id").eq("id", user_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    return r.data[0].get("household_id")


def get_user_role(supabase: Client, user_id: str) -> str | None:
    """Resolve user_id to role (elder, caregiver, admin). Returns None if user not in public.users."""
    r = supabase.table("users").select("role").eq("id", user_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        return None
    return r.data[0].get("role")


def ingest_events(
    body: IngestEventsRequest,
    household_id: str,
    supabase: Client,
) -> IngestEventsResponse:
    """
    Batch ingest event packets. Verifies all session_ids belong to household;
    inserts events and returns counts. Caller must resolve user -> household and
    raise 403 if no household.
    """
    if not body.events:
        return IngestEventsResponse(ingested=0, session_ids=[], last_ts=None)

    def _sid(e):
        return e.session_id if hasattr(e, "session_id") else e.get("session_id")

    session_ids = list({_sid(e) for e in body.events})
    sessions = (
        supabase.table("sessions")
        .select("id, household_id")
        .in_("id", [str(s) for s in session_ids])
        .execute()
    )
    for s in sessions.data or []:
        if s["household_id"] != household_id:
            raise ValueError("Session not in your household")

    def _get(e, key):
        return getattr(e, key, None) if hasattr(e, key) else e.get(key)

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    last_ts = None
    for e in body.events:
        ts = _get(e, "ts")
        ts_str = ts.isoformat() if isinstance(ts, datetime) else ts
        last_ts = ts
        rows.append({
            "session_id": str(_get(e, "session_id")),
            "device_id": str(_get(e, "device_id")),
            "ts": ts_str,
            "seq": _get(e, "seq"),
            "event_type": _get(e, "event_type"),
            "payload": _get(e, "payload") or {},
            "payload_version": _get(e, "payload_version") or 1,
            "ingested_at": now_iso,
        })
    # Idempotent: same (session_id, seq) updates existing row (deterministic re-ingest).
    supabase.table("events").upsert(rows, on_conflict="session_id,seq").execute()
    return IngestEventsResponse(ingested=len(rows), session_ids=session_ids, last_ts=last_ts)
