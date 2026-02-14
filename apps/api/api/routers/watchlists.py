from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import WatchlistItem, WatchlistListResponse

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


@router.get("", response_model=WatchlistListResponse)
def list_watchlists(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Watchlists for device + UI. Pattern, reason, priority, expires_at."""
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    if not u.data:
        return WatchlistListResponse(watchlists=[])
    r = (
        supabase.table("watchlists")
        .select("id, watch_type, pattern, reason, priority, created_at, expires_at")
        .eq("household_id", u.data["household_id"])
        .order("priority", desc=True)
        .execute()
    )
    data = r.data or []
    watchlists = [
        WatchlistItem(
            id=UUID(w["id"]),
            watch_type=w["watch_type"],
            pattern=w.get("pattern") or {},
            reason=w.get("reason"),
            priority=w.get("priority", 0),
            expires_at=datetime.fromisoformat(w["expires_at"].replace("Z", "+00:00")) if w.get("expires_at") else None,
        )
        for w in data
    ]
    return WatchlistListResponse(watchlists=watchlists)
