"""Incident packets API: GET /incident_packets/{id} (bank-ready case file)."""
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import IncidentPacketResponse
from domain.ingest_service import get_household_id

router = APIRouter(prefix="/incident_packets", tags=["incident_packets"])


@router.get("/{packet_id}", response_model=IncidentPacketResponse)
def get_incident_packet(
    packet_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get bank-ready case file by id. Export/download for caregiver."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    r = (
        supabase.table("incident_packets")
        .select("*")
        .eq("id", str(packet_id))
        .eq("household_id", hh_id)
        .single()
        .execute()
    )
    if not r.data:
        raise HTTPException(status_code=404, detail="Incident packet not found")
    row = r.data
    return IncidentPacketResponse(
        id=UUID(row["id"]),
        household_id=UUID(row["household_id"]),
        risk_signal_id=UUID(row["risk_signal_id"]),
        packet_json=row.get("packet_json") or {},
        created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
    )
