"""Watchlist domain: list watchlists for household."""
from datetime import datetime
from uuid import UUID

from supabase import Client

from api.schemas import WatchlistItem, WatchlistListResponse


def list_watchlists(
    household_id: str,
    supabase: Client,
) -> WatchlistListResponse:
    """List watchlists for household (pattern, reason, priority, expires_at)."""
    if not household_id:
        return WatchlistListResponse(watchlists=[])
    r = (
        supabase.table("watchlists")
        .select("id, watch_type, pattern, reason, priority, created_at, expires_at")
        .eq("household_id", household_id)
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
            model_available=True if w.get("watch_type") == "embedding_centroid" else None,
        )
        for w in data
    ]
    return WatchlistListResponse(watchlists=watchlists)
