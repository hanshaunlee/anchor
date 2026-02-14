"""Pydantic schemas for API request/response and UI contracts.

Event packet models mirror docs/event_packet_spec.md as the contract of record.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


# --- Event packet (contract: docs/event_packet_spec.md) ---

# Supported payload schema versions; reject unsupported versions when strict.
SUPPORTED_PAYLOAD_VERSIONS: tuple[int, ...] = (1,)

EventTypeLiteral = Literal[
    "wake", "partial_asr", "final_asr", "intent",
    "tool_call", "tool_result", "tts", "error",
    "device_state", "watchlist_hit",
    "transaction_detected", "payee_added", "bank_alert_received",
]


class SpeakerRole(str, Enum):
    elder = "elder"
    agent = "agent"
    unknown = "unknown"


class SpeakerPayload(BaseModel):
    """Nested speaker: speaker_id?, role? (elder|agent|unknown)."""
    model_config = ConfigDict(extra="forbid")
    speaker_id: str | None = None
    role: SpeakerRole | None = None


class FinalAsrPayload(BaseModel):
    """final_asr payload: text?, text_hash?, lang?, confidence?, speaker?."""
    model_config = ConfigDict(extra="forbid")
    text: str | None = None
    text_hash: str | None = None
    lang: str | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    speaker: SpeakerPayload | None = None


class IntentPayload(BaseModel):
    """intent payload: name required, slots?, confidence?."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1)
    slots: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(None, ge=0.0, le=1.0)


class DeviceStatePayload(BaseModel):
    """device_state payload: online required, battery?, wifi_ssid_hash?."""
    model_config = ConfigDict(extra="forbid")
    online: bool = ...
    battery: float | None = Field(None, ge=0.0, le=1.0)
    wifi_ssid_hash: str | None = None


class EventPacket(BaseModel):
    """
    Single event packet. Contract: docs/event_packet_spec.md.
    Strict validation; payload shape validated by event_type where defined.
    """
    model_config = ConfigDict(strict=True, extra="forbid")

    session_id: UUID
    device_id: UUID
    ts: datetime
    seq: int = Field(..., ge=0)
    event_type: EventTypeLiteral
    payload_version: int = Field(1, description="Version of payload schema")
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload_version")
    @classmethod
    def payload_version_supported(cls, v: int) -> int:
        if v not in SUPPORTED_PAYLOAD_VERSIONS:
            raise ValueError(f"payload_version {v} not in supported {SUPPORTED_PAYLOAD_VERSIONS}")
        return v

    @model_validator(mode="after")
    def validate_payload_by_event_type(self) -> "EventPacket":
        """Strict payload validation for known event types."""
        if not self.payload:
            return self
        et = self.event_type
        if et == "final_asr":
            FinalAsrPayload.model_validate(self.payload)
        elif et == "intent":
            IntentPayload.model_validate(self.payload)
        elif et == "device_state":
            DeviceStatePayload.model_validate(self.payload)
        # wake, partial_asr, tool_call, tool_result, tts, error, watchlist_hit,
        # transaction_detected, payee_added, bank_alert_received: allow dict
        return self


# Backward compatibility: ingest still accepts the same request shape.
IngestEventItem = EventPacket


class IngestEventsRequest(BaseModel):
    """Raw events list; each item validated to EventPacket in handler for rejection logging."""
    events: list[dict[str, Any]]


class IngestEventsResponse(BaseModel):
    ingested: int
    session_ids: list[UUID]
    last_ts: datetime | None
    rejected: int = 0
    rejection_reasons: list[str] = Field(default_factory=list)


# --- Household ---
class HouseholdMe(BaseModel):
    id: UUID
    name: str
    role: UserRole
    display_name: str | None


class ConsentDefaults(BaseModel):
    share_with_caregiver: bool | None = None
    share_text: bool | None = None
    allow_outbound_contact: bool | None = None
    escalation_threshold: int | None = Field(None, ge=1, le=5)


class CaregiverContactCreate(BaseModel):
    name: str
    relationship: str | None = None
    phone: str | None = None
    email: str | None = None
    quiet_hours: dict[str, Any] | None = None
    priority: int = 1


class OnboardRequest(BaseModel):
    """Body for POST /households/onboard: optional display name, household name, consent defaults."""
    display_name: str | None = None
    household_name: str | None = None
    consent_defaults: ConsentDefaults | None = None


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
    model_available: bool | None = Field(None, description="True when GNN produced this signal; false or null when rule-only")


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


# --- Single risk scoring contract (pipeline, worker, agent) ---
class RiskScoreItem(BaseModel):
    """One risk score from the shared scoring service. model_available and embedding/model_subgraph presence are explicit."""
    node_type: str = "entity"
    node_index: int = Field(..., ge=0)
    score: float = Field(..., ge=0.0, le=1.0)
    signal_type: str | None = Field("relational_anomaly", description="e.g. relational_anomaly")
    embedding: list[float] | None = Field(None, description="Present only when model ran and returned embeddings")
    model_subgraph: dict[str, Any] | None = Field(None, description="Present only when model ran and explainer produced subgraph")
    model_available: bool = Field(False, description="True when this score came from the GNN; false for rule-only fallback")


class RiskScoringModelMeta(BaseModel):
    """Model provenance when GNN ran (for similar incidents, centroid watchlists)."""
    model_name: str | None = None
    checkpoint_id: str | None = None
    embedding_dim: int | None = None


class RiskScoringResponse(BaseModel):
    """Single contract for risk scoring used by pipeline, worker, and Financial Security Agent. No silent placeholders."""
    model_available: bool = Field(..., description="True when GNN ran and produced scores; false when model unavailable")
    scores: list[RiskScoreItem] = Field(default_factory=list)
    fallback_used: str | None = Field(None, description="e.g. 'rule_only' when caller added explicit rule-based scores after model_available=false")
    model_meta: RiskScoringModelMeta | None = Field(None, description="Set when model_available=True; model_name, checkpoint_id, embedding_dim")


class FeedbackSubmit(BaseModel):
    label: FeedbackLabel
    notes: str | None = None


# --- Watchlists ---
class EmbeddingCentroidProvenance(BaseModel):
    """Provenance for embedding_centroid watchlist (GNN-dependent)."""
    risk_signal_ids: list[str] = Field(default_factory=list)
    window_days: int | None = None


class EmbeddingCentroidPattern(BaseModel):
    """watch_type=embedding_centroid: L2-normalized centroid, cosine threshold. Hard dependency: no centroid when model didn't run."""
    metric: str = "cosine"
    threshold: float = Field(..., description="cosine similarity threshold (e.g. 0.82)")
    centroid: list[float] = Field(..., description="L2-normalized embedding vector")
    dim: int = Field(..., ge=1)
    model_name: str | None = None
    provenance: EmbeddingCentroidProvenance | None = None


class WatchlistItem(BaseModel):
    id: UUID
    watch_type: str
    pattern: dict[str, Any]
    reason: str | None
    priority: int
    expires_at: datetime | None
    model_available: bool | None = Field(None, description="True when watch_type=embedding_centroid (GNN-dependent); false/null for hash/keyword")


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
    high_risk_mode: dict[str, Any] | None = Field(None, description="When playbook pushed device_high_risk_mode; device enacts")


# --- Weekly summary ---
class SimilarIncident(BaseModel):
    """One similar past incident for Similar Incidents panel (real embeddings only)."""
    risk_signal_id: UUID
    similarity: float = Field(description="Cosine similarity from real embedding")
    score: float = Field(description="Same as similarity (backward compat)")
    ts: datetime | None = None
    signal_type: str | None = None
    severity: int | None = None
    status: str | None = None
    label_outcome: str | None = Field(None, description="confirmed_scam | false_positive | open")
    outcome: str | None = Field(None, description="Same as label_outcome")


class RetrievalProvenance(BaseModel):
    """Provenance for similar-incidents retrieval (model_available path only)."""
    model_name: str | None = None
    checkpoint_id: str | None = None
    embedding_dim: int | None = None
    timestamp: datetime | None = None


class SimilarIncidentsResponse(BaseModel):
    """When model did not run, available=false and similar=[]; do not compute similarity on synthetic embeddings."""
    available: bool = Field(True, description="False when no embedding (e.g. model_not_run)")
    reason: str | None = Field(None, description="e.g. 'model_not_run' when embedding missing")
    similar: list[SimilarIncident] = Field(default_factory=list)
    retrieval_provenance: RetrievalProvenance | None = Field(None, description="Set when available=True; model_name, checkpoint_id, embedding_dim, timestamp")


class WeeklySummary(BaseModel):
    id: UUID
    period_start: datetime | None
    period_end: datetime | None
    summary_text: str
    summary_json: dict[str, Any]


# --- Outbound outreach (caregiver notify/call/email) ---
class OutreachActionRequest(BaseModel):
    risk_signal_id: UUID
    channel_preference: str | None = Field(None, description="sms | email | voice_call")
    dry_run: bool = Field(False, description="If true, preview only; no send, no DB write")


class OutboundActionPayloadElderSafe(BaseModel):
    """Elder-safe view: only elder_safe_message and status."""
    elder_safe_message: str | None = None
    status: str = "queued"
    created_at: datetime | None = None


class OutboundActionResponse(BaseModel):
    id: UUID
    household_id: UUID
    triggered_by_risk_signal_id: UUID | None
    action_type: str
    channel: str
    recipient_name: str | None
    recipient_contact: str | None = None  # Redacted for elder; caregiver sees full
    recipient_contact_last4: str | None = None  # Safe for display; never raw to elder
    payload: dict[str, Any]
    status: str  # queued | sent | delivered | failed | suppressed
    provider: str
    provider_message_id: str | None
    error: str | None
    created_at: datetime
    sent_at: datetime | None
    delivered_at: datetime | None


class OutreachSummaryCounts(BaseModel):
    sent: int = 0
    suppressed: int = 0
    failed: int = 0
    queued: int = 0
    delivered: int = 0


class OutreachSummaryRecentItem(BaseModel):
    id: UUID
    status: str
    created_at: datetime | None
    sent_at: datetime | None
    error: str | None
    triggered_by_risk_signal_id: UUID | None
    channel: str | None
    recipient_contact_last4: str | None


class OutreachSummaryResponse(BaseModel):
    counts: OutreachSummaryCounts
    recent: list[OutreachSummaryRecentItem]


# --- Capabilities (household) ---
class CapabilitiesMe(BaseModel):
    household_id: UUID
    notify_sms_enabled: bool
    notify_email_enabled: bool
    device_policy_push_enabled: bool
    bank_data_connector: str
    bank_control_capabilities: dict[str, Any]
    updated_at: datetime | None


class CapabilitiesPatch(BaseModel):
    notify_sms_enabled: bool | None = None
    notify_email_enabled: bool | None = None
    device_policy_push_enabled: bool | None = None
    bank_data_connector: str | None = None
    bank_control_capabilities: dict[str, Any] | None = None


# --- Action playbooks & tasks ---
class ActionTaskDetail(BaseModel):
    id: UUID
    playbook_id: UUID
    task_type: str
    status: str
    details: dict[str, Any]  # API redacts phone/email for elder
    completed_by_user_id: UUID | None
    completed_at: datetime | None
    created_at: datetime


class PlaybookDetail(BaseModel):
    id: UUID
    household_id: UUID
    risk_signal_id: UUID
    playbook_type: str
    graph: dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime
    tasks: list[ActionTaskDetail] = Field(default_factory=list)


class PlaybookTaskCompleteRequest(BaseModel):
    notes: str | None = None


# --- Incident packets ---
class IncidentPacketResponse(BaseModel):
    id: UUID
    household_id: UUID
    risk_signal_id: UUID
    packet_json: dict[str, Any]
    created_at: datetime


# --- Incident response agent ---
class IncidentResponseRunRequest(BaseModel):
    risk_signal_id: UUID = Field(..., description="Risk signal to run incident response for")
    dry_run: bool = Field(False, description="If true, no DB write; return preview")
