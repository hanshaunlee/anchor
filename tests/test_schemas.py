"""Tests for api.schemas: Pydantic model validation."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from api.schemas import (
    UserRole,
    SessionMode,
    RiskSignalStatus,
    FeedbackLabel,
    IngestEventItem,
    IngestEventsRequest,
    RiskSignalCard,
    RiskSignalDetail,
    FeedbackSubmit,
    WatchlistItem,
    SimilarIncident,
)


def test_ingest_event_item_valid() -> None:
    ts = datetime.now(timezone.utc)
    sid = uuid4()
    did = uuid4()
    item = IngestEventItem(
        session_id=sid,
        device_id=did,
        ts=ts,
        seq=0,
        event_type="final_asr",
        payload={"text": "hello"},
    )
    assert item.session_id == sid
    assert item.event_type == "final_asr"
    assert item.payload == {"text": "hello"}
    assert item.payload_version == 1


def test_ingest_event_item_default_payload() -> None:
    item = IngestEventItem(
        session_id=uuid4(),
        device_id=uuid4(),
        ts=datetime.now(timezone.utc),
        seq=0,
        event_type="wake",
    )
    assert item.payload == {}


def test_ingest_events_request() -> None:
    req = IngestEventsRequest(events=[])
    assert req.events == []
    req = IngestEventsRequest(
        events=[
            IngestEventItem(
                session_id=uuid4(),
                device_id=uuid4(),
                ts=datetime.now(timezone.utc),
                seq=0,
                event_type="intent",
                payload={"name": "call"},
            )
        ]
    )
    assert len(req.events) == 1


def test_risk_signal_card_severity_bounds() -> None:
    RiskSignalCard(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        signal_type="test",
        severity=3,
        score=0.5,
        status=RiskSignalStatus.open,
    )
    with pytest.raises(ValidationError):
        RiskSignalCard(
            id=uuid4(),
            ts=datetime.now(timezone.utc),
            signal_type="test",
            severity=0,  # invalid
            score=0.5,
            status=RiskSignalStatus.open,
        )
    with pytest.raises(ValidationError):
        RiskSignalCard(
            id=uuid4(),
            ts=datetime.now(timezone.utc),
            signal_type="test",
            severity=6,  # invalid
            score=0.5,
            status=RiskSignalStatus.open,
        )


def test_feedback_submit() -> None:
    body = FeedbackSubmit(label=FeedbackLabel.true_positive, notes="looks like scam")
    assert body.label == FeedbackLabel.true_positive
    assert body.notes == "looks like scam"
    body2 = FeedbackSubmit(label=FeedbackLabel.false_positive)
    assert body2.notes is None


def test_watchlist_item() -> None:
    w = WatchlistItem(
        id=uuid4(),
        watch_type="entity_pattern",
        pattern={"entity_index": 0},
        reason="High risk",
        priority=1,
        expires_at=None,
    )
    assert w.watch_type == "entity_pattern"
    assert w.priority == 1


def test_similar_incident() -> None:
    s = SimilarIncident(
        risk_signal_id=uuid4(),
        score=0.9,
        outcome="confirmed_scam",
        ts=datetime.now(timezone.utc),
    )
    assert s.score == 0.9
    assert s.outcome == "confirmed_scam"


def test_enums() -> None:
    assert UserRole.elder.value == "elder"
    assert SessionMode.offline.value == "offline"
    assert RiskSignalStatus.escalated.value == "escalated"
