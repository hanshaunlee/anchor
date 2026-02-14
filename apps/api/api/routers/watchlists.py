from fastapi import APIRouter, Depends
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import WatchlistListResponse
from domain.ingest_service import get_household_id
from domain.watchlist_service import list_watchlists

router = APIRouter(prefix="/watchlists", tags=["watchlists"])


@router.get("", response_model=WatchlistListResponse)
def list_watchlists_route(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Watchlists for device + UI. Pattern, reason, priority, expires_at."""
    hh_id = get_household_id(supabase, user_id)
    return list_watchlists(hh_id or "", supabase)
