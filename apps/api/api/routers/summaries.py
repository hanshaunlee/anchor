from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import WeeklySummary

router = APIRouter(prefix="/summaries", tags=["summaries"])


@router.get("", response_model=list[WeeklySummary])
def list_summaries(
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    session_id: UUID | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Weekly rollups and session summaries. UI: period_start/period_end for weekly view."""
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    if not u.data:
        return []
    q = (
        supabase.table("summaries")
        .select("id, period_start, period_end, summary_text, summary_json, created_at")
        .eq("household_id", u.data["household_id"])
        .order("period_start", desc=True)
        .limit(min(limit * 3, 100))  # fetch extra so we can prefer period-scoped below
    )
    if from_ts:
        q = q.gte("period_end", from_ts.isoformat())
    if to_ts:
        q = q.lte("period_start", to_ts.isoformat())
    if session_id:
        q = q.eq("session_id", str(session_id))
    r = q.execute()
    raw = r.data or []
    # Prefer period-scoped (weekly) summaries first; then session-scoped. Within period-scoped, newest first.
    period_scoped = [s for s in raw if s.get("period_start")]
    session_scoped = [s for s in raw if not s.get("period_start")]
    period_scoped.sort(key=lambda s: s.get("period_start") or "", reverse=True)
    data = (period_scoped + session_scoped)[:limit]
    return [
        WeeklySummary(
            id=UUID(s["id"]),
            period_start=datetime.fromisoformat(s["period_start"].replace("Z", "+00:00")) if s.get("period_start") else None,
            period_end=datetime.fromisoformat(s["period_end"].replace("Z", "+00:00")) if s.get("period_end") else None,
            summary_text=s.get("summary_text") or "",
            summary_json=s.get("summary_json") or {},
        )
        for s in data
    ]
