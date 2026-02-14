"""Unit tests for domain.graph_service: build_graph_from_events and normalize_events (single place for graph build)."""
import pytest

from domain.graph_service import build_graph_from_events, normalize_events


def test_normalize_events_returns_four_lists() -> None:
    """normalize_events(household_id, events) returns (utterances, entities, mentions, relationships)."""
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "Call Mom", "confidence": 0.9}},
    ]
    utterances, entities, mentions, relationships = normalize_events("hh1", events)
    assert isinstance(utterances, list)
    assert isinstance(entities, list)
    assert isinstance(mentions, list)
    assert isinstance(relationships, list)


def test_build_graph_from_events_no_supabase_same_as_normalize() -> None:
    """build_graph_from_events(household_id, events, supabase=None) returns same shape as normalize_events."""
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "Hello", "confidence": 0.9}},
    ]
    u1, e1, m1, r1 = normalize_events("hh1", events)
    u2, e2, m2, r2 = build_graph_from_events("hh1", events, supabase=None)
    assert len(u1) == len(u2)
    assert len(e1) == len(e2)
    assert len(m1) == len(m2)
    assert len(r1) == len(r2)


def test_build_graph_from_events_empty_events() -> None:
    """build_graph_from_events with empty events returns empty lists."""
    u, e, m, r = build_graph_from_events("hh1", [], supabase=None)
    assert u == []
    assert e == []
    assert m == []
    assert r == []
