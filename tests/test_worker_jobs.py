"""Tests for apps/worker/worker/jobs: run_graph_builder, run_risk_inference, ingest_events_batch."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from worker.jobs import (
    ingest_events_batch,
    run_graph_builder,
    run_risk_inference,
    run_pipeline,
)


def test_ingest_events_batch_returns_list_from_execute() -> None:
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    q = mock_supabase.table.return_value
    q.select.return_value = q
    q.eq.return_value = q
    q.gte.return_value = q
    q.lte.return_value = q
    events = ingest_events_batch(mock_supabase, "hh1", None, None)
    assert events == []


def test_ingest_events_batch_calls_supabase() -> None:
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value.data = [
        {"id": "e1", "session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
    ]
    mock_supabase.table.return_value.eq.return_value.execute.return_value.data = []
    # Chain: table("events").select(...).eq(...).gte(...).lte(...).execute()
    q = mock_supabase.table.return_value
    q.select.return_value = q
    q.eq.return_value = q
    q.gte.return_value = q
    q.lte.return_value = q
    q.execute.return_value.data = [{"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}}]
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    events = ingest_events_batch(mock_supabase, "hh1", start, end)
    assert len(events) == 1
    assert events[0]["event_type"] == "wake"


def test_run_graph_builder() -> None:
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "Call John", "confidence": 0.9}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:01Z", "seq": 2, "event_type": "intent", "payload": {"name": "call", "slots": {"contact": "John"}, "confidence": 0.85}},
    ]
    result = run_graph_builder(None, "hh1", events)
    assert "entities" in result
    assert "mentions" in result
    assert "relationships" in result
    assert "utterances" in result
    assert len(result["utterances"]) >= 1


def test_run_risk_inference() -> None:
    graph_data = {
        "entities": [{"id": "e1"}, {"id": "e2"}],
        "mentions": [],
        "relationships": [],
        "utterances": [],
    }
    scores = run_risk_inference("hh1", graph_data)
    assert len(scores) == 2
    for s in scores:
        assert s["household_id"] == "hh1"
        assert "severity" in s
        assert "score" in s
        assert s["signal_type"] == "relational_anomaly"
        assert "explanation" in s


def test_run_pipeline_no_supabase() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("langgraph")
    result = run_pipeline(None, "hh1", None, None)
    assert "risk_scores" in result
    assert "watchlists" in result
    assert "normalized" in result
    assert result.get("normalized") is True
