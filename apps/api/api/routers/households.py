import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import Client

from api.deps import get_supabase, get_current_user_id, require_user, require_caregiver_or_admin
from api.schemas import CaregiverContactCreate, ConsentDefaults, HouseholdMe, OnboardRequest, UserRole

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
        # Optional: household consent defaults
        if body.consent_defaults:
            cd = body.consent_defaults
            supabase.table("household_consent_defaults").upsert({
                "household_id": h_id,
                "share_with_caregiver": cd.share_with_caregiver if cd.share_with_caregiver is not None else True,
                "share_text": cd.share_text if cd.share_text is not None else True,
                "allow_outbound_contact": cd.allow_outbound_contact if cd.allow_outbound_contact is not None else False,
                "escalation_threshold": cd.escalation_threshold if cd.escalation_threshold is not None else 3,
                "updated_at": now,
            }, on_conflict="household_id").execute()
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


def _get_consent_for_household(supabase: Client, hh_id: str) -> dict:
    d = supabase.table("household_consent_defaults").select("*").eq("household_id", hh_id).limit(1).execute()
    if not d.data or len(d.data) == 0:
        return {
            "share_with_caregiver": True,
            "share_text": True,
            "allow_outbound_contact": False,
            "escalation_threshold": 3,
        }
    row = d.data[0]
    return {
        "share_with_caregiver": row.get("share_with_caregiver", True),
        "share_text": row.get("share_text", True),
        "allow_outbound_contact": row.get("allow_outbound_contact", False),
        "escalation_threshold": row.get("escalation_threshold", 3),
        "updated_at": row.get("updated_at"),
    }


@router.get("/me/consent")
def get_household_consent(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get household consent defaults. Per-session overrides remain in sessions.consent_state."""
    r = supabase.table("users").select("household_id").eq("id", user_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Not onboarded")
    return _get_consent_for_household(supabase, r.data[0]["household_id"])


class ConsentPatch(BaseModel):
    share_with_caregiver: bool | None = None
    share_text: bool | None = None
    allow_outbound_contact: bool | None = None
    escalation_threshold: int | None = None


@router.patch("/me/consent")
def patch_household_consent(
    body: ConsentPatch,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Update household consent defaults (elder or caregiver)."""
    r = supabase.table("users").select("household_id").eq("id", user_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Not onboarded")
    hh_id = r.data[0]["household_id"]
    now = datetime.now(timezone.utc).isoformat()
    payload = body.model_dump(exclude_none=True)
    if not payload:
        return get_household_consent(user_id, supabase)
    payload["updated_at"] = now
    supabase.table("household_consent_defaults").upsert(
        {"household_id": hh_id, **payload},
        on_conflict="household_id",
    ).execute()
    return _get_consent_for_household(supabase, hh_id)


@router.get("/me/contacts")
def list_caregiver_contacts(
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """List caregiver contacts (caregiver/admin only). Elder cannot see contact details."""
    r = supabase.table("users").select("household_id").eq("id", user_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Not onboarded")
    hh_id = r.data[0]["household_id"]
    c = supabase.table("caregiver_contacts").select("*").eq("household_id", hh_id).order("priority").execute()
    return {"contacts": c.data or []}


@router.post("/me/contacts")
def add_caregiver_contact(
    body: CaregiverContactCreate,
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """Add a caregiver contact (caregiver/admin only)."""
    r = supabase.table("users").select("household_id").eq("id", user_id).limit(1).execute()
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Not onboarded")
    hh_id = r.data[0]["household_id"]
    now = datetime.now(timezone.utc).isoformat()
    channels = {}
    if body.phone:
        channels["phone"] = body.phone
    if body.email:
        channels["email"] = body.email
    supabase.table("caregiver_contacts").insert({
        "household_id": hh_id,
        "user_id": user_id,
        "name": body.name,
        "relationship": body.relationship,
        "channels": channels,
        "priority": body.priority,
        "quiet_hours": body.quiet_hours or {},
        "verified": False,
        "created_at": now,
        "updated_at": now,
    }).execute()
    return {"ok": True}
