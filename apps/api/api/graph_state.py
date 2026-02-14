"""
LangGraph state for Anchor pipeline.
Ingest -> Normalize -> GraphUpdate -> RiskScore -> Explain -> ConsentGate -> WatchlistSynthesis -> EscalationDraft -> Persist.
"""
from typing import Annotated, Any, Literal

from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AnchorState(BaseModel):
    """State passed through the pipeline; checkpointed for durability."""

    household_id: str = ""
    time_range_start: str | None = None
    time_range_end: str | None = None
    ingested_events: list[dict[str, Any]] = Field(default_factory=list)
    session_ids: list[str] = Field(default_factory=list)
    normalized: bool = False
    utterances: list[dict] = Field(default_factory=list)
    entities: list[dict] = Field(default_factory=list)
    mentions: list[dict] = Field(default_factory=list)
    relationships: list[dict] = Field(default_factory=list)
    graph_updated: bool = False
    risk_scores: list[dict] = Field(default_factory=list)
    explanations: list[dict] = Field(default_factory=list)
    consent_allows_escalation: bool = True
    consent_allows_watchlist: bool = True
    watchlists: list[dict] = Field(default_factory=list)
    escalation_draft: str = ""
    persisted: bool = False
    needs_review: bool = False
    severity_threshold: int = 4
    consent_state: dict[str, Any] = Field(default_factory=dict)  # share_with_caregiver, etc.
    time_to_flag: float | None = None  # seconds from first event to first score above threshold
    logs: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


def append_log(state: dict, msg: str) -> dict:
    state["logs"] = state.get("logs", []) + [msg]
    return state
