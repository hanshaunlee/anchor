"""Capabilities API: GET /capabilities/me, PATCH /capabilities (caregiver/admin)."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from uuid import UUID

from api.deps import get_supabase, require_user, require_caregiver_or_admin
from api.schemas import CapabilitiesMe, CapabilitiesPatch
from domain.capability_service import get_household_capabilities, update_household_capabilities
from domain.ingest_service import get_household_id

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


@router.get("/me", response_model=CapabilitiesMe)
def get_capabilities_me(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Return current household capabilities. Elder can view; caregiver/admin can also PATCH."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    cap = get_household_capabilities(supabase, hh_id)
    return CapabilitiesMe(
        household_id=UUID(cap["household_id"]) if isinstance(cap["household_id"], str) else cap["household_id"],
        notify_sms_enabled=cap.get("notify_sms_enabled", False),
        notify_email_enabled=cap.get("notify_email_enabled", False),
        device_policy_push_enabled=cap.get("device_policy_push_enabled", True),
        bank_data_connector=cap.get("bank_data_connector", "none"),
        bank_control_capabilities=cap.get("bank_control_capabilities") or {},
        updated_at=datetime.fromisoformat(cap["updated_at"].replace("Z", "+00:00")) if cap.get("updated_at") else None,
    )


@router.patch("", response_model=CapabilitiesMe)
def patch_capabilities(
    body: CapabilitiesPatch,
    user_id: str = Depends(require_caregiver_or_admin),
    supabase: Client = Depends(get_supabase),
):
    """Update household capabilities (caregiver/admin only). For demo config."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    patch = body.model_dump(exclude_none=True)
    cap = update_household_capabilities(supabase, hh_id, patch)
    return CapabilitiesMe(
        household_id=UUID(cap["household_id"]) if isinstance(cap["household_id"], str) else cap["household_id"],
        notify_sms_enabled=cap.get("notify_sms_enabled", False),
        notify_email_enabled=cap.get("notify_email_enabled", False),
        device_policy_push_enabled=cap.get("device_policy_push_enabled", True),
        bank_data_connector=cap.get("bank_data_connector", "none"),
        bank_control_capabilities=cap.get("bank_control_capabilities") or {},
        updated_at=datetime.fromisoformat(cap["updated_at"].replace("Z", "+00:00")) if cap.get("updated_at") else None,
    )
