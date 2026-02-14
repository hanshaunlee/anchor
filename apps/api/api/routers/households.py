import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, get_current_user_id, require_user
from api.schemas import HouseholdMe, OnboardRequest, UserRole

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/households", tags=["households"])


@router.post("/onboard", response_model=HouseholdMe)
def onboard_household(
    body: OnboardRequest | None = None,
    user_id: str | None = Depends(get_current_user_id),
    supabase: Client = Depends(get_supabase),
):
    """
    Create a household and link the authenticated user (no row in public.users yet).
    Call after sign-up so the user gets a household and can use the API. Idempotent: if user
    already has a household, returns current HouseholdMe.
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    body = body or OnboardRequest()
    # Already onboarded?
    r = supabase.table("users").select("household_id, role, display_name").eq("id", user_id).limit(1).execute()
    if r.data and len(r.data) > 0:
        h = supabase.table("households").select("id, name").eq("id", r.data["household_id"]).single().execute()
        if h.data:
            return HouseholdMe(
                id=UUID(h.data["id"]),
                name=h.data["name"],
                role=UserRole(r.data["role"]),
                display_name=r.data.get("display_name"),
            )
    # Create household then user (service role bypasses RLS)
    now = datetime.now(timezone.utc).isoformat()
    h_name = (body.household_name or "").strip() or "My Household"
    try:
        h_res = supabase.table("households").insert({"name": h_name}).execute()
        if not h_res.data or len(h_res.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create household")
        h_id = h_res.data[0]["id"]
        u_res = supabase.table("users").insert({
            "id": user_id,
            "household_id": h_id,
            "role": "caregiver",
            "display_name": (body.display_name or "").strip() or None,
            "created_at": now,
            "updated_at": now,
        }).execute()
        if not u_res.data or len(u_res.data) == 0:
            logger.error("Onboard: users insert returned no data for user_id=%s", user_id)
            raise HTTPException(status_code=500, detail="Database error saving new user")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Onboard: database error: %s", e)
        raise HTTPException(status_code=500, detail="Database error saving new user") from e
    return HouseholdMe(
        id=UUID(h_id),
        name=h_name,
        role=UserRole.caregiver,
        display_name=(body.display_name or "").strip() or None,
    )


@router.get("/me", response_model=HouseholdMe)
def get_household_me(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Return current user's household metadata and role. UI: use for nav and role-based views.
    Returns 404 if user has no row in public.users (e.g. signed up but not onboarded)."""
    r = (
        supabase.table("users")
        .select("household_id, role, display_name")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Not onboarded. Call POST /households/onboard to create a household.")
    user_row = r.data[0]
    h = (
        supabase.table("households")
        .select("id, name")
        .eq("id", user_row["household_id"])
        .limit(1)
        .execute()
    )
    if not h.data or len(h.data) == 0:
        raise HTTPException(status_code=404, detail="Household not found")
    h_row = h.data[0]
    return HouseholdMe(
        id=UUID(h_row["id"]),
        name=h_row["name"],
        role=UserRole(user_row["role"]),
        display_name=user_row.get("display_name"),
    )
