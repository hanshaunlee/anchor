from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import HouseholdMe, UserRole

router = APIRouter(prefix="/households", tags=["households"])


@router.get("/me", response_model=HouseholdMe)
def get_household_me(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Return current user's household metadata and role. UI: use for nav and role-based views."""
    r = (
        supabase.table("users")
        .select("household_id, role, display_name")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="User not found")
    h = (
        supabase.table("households")
        .select("id, name")
        .eq("id", r.data["household_id"])
        .single()
        .execute()
    )
    if not h.data:
        raise HTTPException(status_code=404, detail="Household not found")
    return HouseholdMe(
        id=UUID(h.data["id"]),
        name=h.data["name"],
        role=UserRole(r.data["role"]),
        display_name=r.data.get("display_name"),
    )
