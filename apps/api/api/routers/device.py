from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import DeviceSyncRequest, DeviceSyncResponse, WatchlistItem

router = APIRouter(prefix="/device", tags=["device"])


@router.post("/sync", response_model=DeviceSyncResponse)
def device_sync(
    body: DeviceSyncRequest,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Device heartbeat: report last_upload_ts and seq by session; returns watchlists delta and last pointers."""
    # Verify device belongs to user's household
    d = (
        supabase.table("devices")
        .select("id, household_id")
        .eq("id", str(body.device_id))
        .single()
        .execute()
    )
    if not d.data:
        raise HTTPException(status_code=404, detail="Device not found")
    u = supabase.table("users").select("household_id").eq("id", user_id).single().execute()
    if not u.data or u.data["household_id"] != d.data["household_id"]:
        raise HTTPException(status_code=403, detail="Device not in your household")

    now = datetime.utcnow().isoformat() + "Z"
    # Upsert device_sync_state
    supabase.table("device_sync_state").upsert(
        {
            "device_id": str(body.device_id),
            "last_upload_ts": body.last_upload_ts.isoformat() if body.last_upload_ts else None,
            "last_upload_seq_by_session": body.last_upload_seq_by_session,
            "last_watchlist_pull_at": now,
            "updated_at": now,
        },
        on_conflict="device_id",
    ).execute()

    # Read back state
    state = (
        supabase.table("device_sync_state")
        .select("last_upload_ts, last_upload_seq_by_session, last_watchlist_pull_at")
        .eq("device_id", str(body.device_id))
        .single()
        .execute()
    )
    data = state.data or {}
    # Watchlists for household (delta = all active; UI/device can filter by last_watchlist_pull_at if needed)
    wl = (
        supabase.table("watchlists")
        .select("id, watch_type, pattern, reason, priority, expires_at")
        .eq("household_id", d.data["household_id"])
        .order("priority", desc=True)
        .execute()
    )
    watchlists = [
        WatchlistItem(
            id=UUID(w["id"]),
            watch_type=w["watch_type"],
            pattern=w.get("pattern") or {},
            reason=w.get("reason"),
            priority=w.get("priority", 0),
            expires_at=datetime.fromisoformat(w["expires_at"].replace("Z", "+00:00")) if w.get("expires_at") else None,
        )
        for w in (wl.data or [])
    ]
    return DeviceSyncResponse(
        watchlists_delta=watchlists,
        last_upload_ts=datetime.fromisoformat(data["last_upload_ts"].replace("Z", "+00:00")) if data.get("last_upload_ts") else None,
        last_upload_seq_by_session=data.get("last_upload_seq_by_session") or {},
        last_watchlist_pull_at=datetime.fromisoformat(data["last_watchlist_pull_at"].replace("Z", "+00:00")) if data.get("last_watchlist_pull_at") else None,
    )
