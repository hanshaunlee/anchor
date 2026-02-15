"""Connectors API: Plaid link_token, exchange_public_token, sync_transactions. 501 when not configured."""
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from api.deps import get_supabase, require_user
from domain.ingest_service import get_household_id

router = APIRouter(prefix="/connectors", tags=["connectors"])


def _plaid_configured() -> bool:
    return bool(os.environ.get("PLAID_CLIENT_ID") and os.environ.get("PLAID_SECRET"))


@router.get("/plaid/link_token")
def plaid_link_token(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Return Plaid Link token for client-side link flow. 501 if Plaid not configured."""
    if not _plaid_configured():
        raise HTTPException(
            status_code=501,
            detail="Plaid not configured. Set PLAID_CLIENT_ID and PLAID_SECRET. See README_EXTENDED.md § 7 (Connectors).",
        )
    # Stub: would call Plaid /link/token/create
    return {"link_token": "stub-not-implemented", "reason": "Plaid stub; implement with Plaid SDK"}


class ExchangePublicTokenRequest(BaseModel):
    public_token: str = Field(..., description="From Plaid Link onSuccess")


@router.post("/plaid/exchange_public_token")
def plaid_exchange_public_token(
    body: ExchangePublicTokenRequest,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Exchange public token for access_token; store per household. 501 if not configured."""
    if not _plaid_configured():
        raise HTTPException(
            status_code=501,
            detail="Plaid not configured. Set PLAID_CLIENT_ID and PLAID_SECRET. See README_EXTENDED.md § 7 (Connectors).",
        )
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    # Stub: would exchange and store in Vault or env; never store raw in DB
    return {"ok": False, "reason": "Plaid stub; implement exchange and secure storage"}


@router.post("/plaid/sync_transactions")
def plaid_sync_transactions(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Sync transactions from Plaid. 501 if not configured or no access token."""
    if not _plaid_configured():
        raise HTTPException(
            status_code=501,
            detail="Plaid not configured. Set PLAID_CLIENT_ID and PLAID_SECRET. See README_EXTENDED.md § 7 (Connectors).",
        )
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    # Stub: would call Plaid /transactions/sync
    return {"ok": False, "reason": "Plaid stub; implement sync and store in events or cache"}
