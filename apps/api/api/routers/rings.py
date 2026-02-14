"""Rings API: list and get ring details (Ring Discovery agent artifacts)."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from api.deps import get_supabase, require_user
from domain.ingest_service import get_household_id

router = APIRouter(prefix="/rings", tags=["rings"])


class RingMember(BaseModel):
    entity_id: str | None
    role: str | None
    first_seen_at: str | None
    last_seen_at: str | None


class RingItem(BaseModel):
    id: UUID
    household_id: UUID
    created_at: str
    updated_at: str
    score: float
    meta: dict = Field(default_factory=dict)


class RingDetail(RingItem):
    members: list[RingMember] = Field(default_factory=list)


class RingListResponse(BaseModel):
    rings: list[RingItem]


@router.get("", response_model=RingListResponse)
def list_rings(
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """List rings for the current household (Ring Discovery agent output)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("rings")
        .select("id, household_id, created_at, updated_at, score, meta")
        .eq("household_id", hh_id)
        .order("created_at", desc=True)
        .execute()
    )
    data = r.data or []
    rings = [
        RingItem(
            id=UUID(row["id"]),
            household_id=UUID(row["household_id"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            score=float(row.get("score", 0)),
            meta=row.get("meta") or {},
        )
        for row in data
    ]
    return RingListResponse(rings=rings)


@router.get("/{ring_id}", response_model=RingDetail)
def get_ring(
    ring_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get a ring by id with members (RLS: household)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=403, detail="No household")
    r = (
        supabase.table("rings")
        .select("id, household_id, created_at, updated_at, score, meta")
        .eq("id", str(ring_id))
        .eq("household_id", hh_id)
        .limit(1)
        .execute()
    )
    if not r.data or len(r.data) == 0:
        raise HTTPException(status_code=404, detail="Ring not found")
    row = r.data[0]
    members_r = (
        supabase.table("ring_members")
        .select("entity_id, role, first_seen_at, last_seen_at")
        .eq("ring_id", str(ring_id))
        .execute()
    )
    members = [
        RingMember(
            entity_id=str(m["entity_id"]) if m.get("entity_id") else None,
            role=m.get("role"),
            first_seen_at=m.get("first_seen_at"),
            last_seen_at=m.get("last_seen_at"),
        )
        for m in (members_r.data or [])
    ]
    return RingDetail(
        id=UUID(row["id"]),
        household_id=UUID(row["household_id"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=float(row.get("score", 0)),
        meta=row.get("meta") or {},
        members=members,
    )
