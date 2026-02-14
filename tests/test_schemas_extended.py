"""Extended schema validation: SessionListItem, EventsListResponse, DeviceSyncRequest/Response, WeeklySummary, etc."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from api.schemas import (
    SessionMode,
    SessionListItem,
    SessionListResponse,
    EventListItem,
    EventsListResponse,
    RiskSignalDetailSubgraph,
    SubgraphNode,
    SubgraphEdge,
    DeviceSyncRequest,
    DeviceSyncResponse,
    WeeklySummary,
    IngestEventsResponse,
)


def test_session_list_item() -> None:
    now = datetime.now(timezone.utc)
    item = SessionListItem(
        id=uuid4(),
        device_id=uuid4(),
        started_at=now,
        ended_at=now,
        mode=SessionMode.offline,
        consent_state={"share_with_caregiver": True},
        summary_text="Session summary",
    )
    assert item.mode == SessionMode.offline
    assert item.summary_text == "Session summary"


def test_session_list_response() -> None:
    r = SessionListResponse(sessions=[], total=0)
    assert r.total == 0
    assert r.sessions == []


def test_event_list_item() -> None:
    item = EventListItem(
        id=uuid4(),
        ts=datetime.now(timezone.utc),
        seq=0,
        event_type="final_asr",
        payload={"text": "hello"},
        text_redacted=False,
    )
    assert item.event_type == "final_asr"
    assert item.text_redacted is False


def test_events_list_response() -> None:
    r = EventsListResponse(events=[], total=0, next_offset=None)
    assert r.next_offset is None


def test_subgraph_node() -> None:
    n = SubgraphNode(id="e1", type="entity", label="Person", score=0.8)
    assert n.score == 0.8


def test_subgraph_edge() -> None:
    e = SubgraphEdge(src="e1", dst="e2", type="co_occurs", weight=1.0, rank=0)
    assert e.weight == 1.0


def test_risk_signal_detail_subgraph() -> None:
    sub = RiskSignalDetailSubgraph(
        nodes=[SubgraphNode(id="e1", type="entity", label=None, score=0.5)],
        edges=[SubgraphEdge(src="e1", dst="e2", type="co_occurs", weight=None, rank=None)],
    )
    assert len(sub.nodes) == 1
    assert len(sub.edges) == 1


def test_device_sync_request() -> None:
    req = DeviceSyncRequest(
        device_id=uuid4(),
        last_upload_ts=datetime.now(timezone.utc),
        last_upload_seq_by_session={"s1": 5},
    )
    assert len(req.last_upload_seq_by_session) == 1


def test_device_sync_response() -> None:
    r = DeviceSyncResponse(
        watchlists_delta=[],
        last_upload_ts=None,
        last_upload_seq_by_session={},
        last_watchlist_pull_at=None,
    )
    assert r.watchlists_delta == []


def test_weekly_summary() -> None:
    w = WeeklySummary(
        id=uuid4(),
        period_start=datetime.now(timezone.utc),
        period_end=datetime.now(timezone.utc),
        summary_text="Weekly recap",
        summary_json={"key": "value"},
    )
    assert w.summary_text == "Weekly recap"


def test_ingest_events_response() -> None:
    r = IngestEventsResponse(ingested=3, session_ids=[uuid4()], last_ts=datetime.now(timezone.utc))
    assert r.ingested == 3
