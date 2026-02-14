"""Pydantic schemas for API request/response and UI contracts."""
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# --- Enums (match DB) ---
class UserRole(str, Enum):
    elder = "elder"
    caregiver = "caregiver"
    admin = "admin"


class SessionMode(str, Enum):
    offline = "offline"
    online = "online"


class RiskSignalStatus(str, Enum):
    open = "open"
    acknowledged = "acknowledged"
    dismissed = "dismissed"
    escalated = "escalated"


class FeedbackLabel(str, Enum):
    true_positive = "true_positive"
    false_positive = "false_positive"
    unsure = "unsure"


# --- Event packet (ingest) ---
class EventPayloadBase(BaseModel):
    """Base for payload variants; actual payload is free-form JSONB."""


class IngestEventItem(BaseModel):
    session_id: UUID
    device_id: UUID
    ts: datetime
    seq: int
    event_type: str = Field(..., description="e.g. wake, final_asr, intent, tool_call, ...")
    payload_version: int = 1
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestEventsRequest(BaseModel):
    events: list[IngestEventItem]


class IngestEventsResponse(BaseModel):
    ingested: int
    session_ids: list[UUID]
    last_ts: datetime | None


# --- Household ---
class HouseholdMe(BaseModel):
    id: UUID
    name: str
    role: UserRole
    display_name: str | None


# --- Sessions ---
class SessionListItem(BaseModel):
    id: UUID
    device_id: UUID
    started_at: datetime
    ended_at: datetime | None
    mode: SessionMode
    consent_state: dict[str, Any]
    summary_text: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]
    total: int


class EventListItem(BaseModel):
    id: UUID
    ts: datetime
    seq: int
    event_type: str
    payload: dict[str, Any]
    text_redacted: bool


class EventsListResponse(BaseModel):
    events: list[EventListItem]
    total: int
    next_offset: int | None


# --- Risk signals (UI contract) ---
class RiskSignalCard(BaseModel):
    """Compact card for list view."""
    id: UUID
    ts: datetime
    signal_type: str
    severity: int = Field(ge=1, le=5)
    score: float
    status: RiskSignalStatus
    summary: str | None = None


class SubgraphNode(BaseModel):
    id: str
    type: str
    label: str | None = None
    score: float | None = None


class SubgraphEdge(BaseModel):
    src: str
    dst: str
    type: str
    weight: float | None = None
    rank: int | None = None


class RiskSignalDetailSubgraph(BaseModel):
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


class RiskSignalDetail(BaseModel):
    """Full detail for UI: includes explanation subgraph."""
    id: UUID
    household_id: UUID
    ts: datetime
    signal_type: str
    severity: int
    score: float
    status: RiskSignalStatus
    explanation: dict[str, Any]
    recommended_action: dict[str, Any]
    subgraph: RiskSignalDetailSubgraph | None = None
    session_ids: list[UUID] = Field(default_factory=list)
    event_ids: list[UUID] = Field(default_factory=list)
    entity_ids: list[UUID] = Field(default_factory=list)


class RiskSignalListResponse(BaseModel):
    signals: list[RiskSignalCard]
    total: int


class FeedbackSubmit(BaseModel):
    label: FeedbackLabel
    notes: str | None = None


# --- Watchlists ---
class WatchlistItem(BaseModel):
    id: UUID
    watch_type: str
    pattern: dict[str, Any]
    reason: str | None
    priority: int
    expires_at: datetime | None


class WatchlistListResponse(BaseModel):
    watchlists: list[WatchlistItem]


# --- Device sync ---
class DeviceSyncRequest(BaseModel):
    device_id: UUID
    last_upload_ts: datetime | None = None
    last_upload_seq_by_session: dict[str, int] = Field(default_factory=dict)


class DeviceSyncResponse(BaseModel):
    watchlists_delta: list[WatchlistItem]
    last_upload_ts: datetime | None
    last_upload_seq_by_session: dict[str, int]
    last_watchlist_pull_at: datetime | None


# --- Weekly summary ---
class SimilarIncident(BaseModel):
    """One similar past incident for Similar Incidents panel."""
    risk_signal_id: UUID
    score: float = Field(description="Cosine similarity or score")
    outcome: str | None = Field(None, description="confirmed_scam | false_positive | open")
    ts: datetime | None = None


class SimilarIncidentsResponse(BaseModel):
    similar: list[SimilarIncident]


class WeeklySummary(BaseModel):
    id: UUID
    period_start: datetime | None
    period_end: datetime | None
    summary_text: str
    summary_json: dict[str, Any]
