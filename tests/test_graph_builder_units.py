"""Unit tests for ml.graph.builder: GraphBuilder edge cases, get_relationship_list, co-occurrence."""
import pytest

pytest.importorskip("torch")
from ml.graph.builder import GraphBuilder, build_hetero_from_tables


def test_graph_builder_empty_events() -> None:
    b = GraphBuilder("hh1")
    b.process_events([], "s1", "d1")
    assert b.get_utterance_list() == []
    assert b.get_entity_list() == []
    assert b.get_mention_list() == []
    assert b.get_relationship_list() == []


def test_graph_builder_get_mention_list() -> None:
    """Mentions are created for entities extracted from intent slots; each has session_id, entity_id, ts."""
    b = GraphBuilder("hh1")
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "intent", "payload": {"name": "call", "slots": {"name": "Alice"}, "confidence": 0.9}},
    ]
    b.process_events(events, "s1", "d1")
    mentions = b.get_mention_list()
    assert isinstance(mentions, list)
    assert len(mentions) >= 1
    for m in mentions:
        assert "session_id" in m
        assert "entity_id" in m
        assert m["session_id"] == "s1"
        assert m["entity_id"].startswith("entity_")


def test_graph_builder_intent_only_creates_entities() -> None:
    b = GraphBuilder("hh1")
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "intent", "payload": {"name": "call", "slots": {"contact": "John"}, "confidence": 0.9}},
    ]
    b.process_events(events, "s1", "d1")
    entities = b.get_entity_list()
    assert len(entities) >= 1
    assert any(e.get("entity_type") in ("person", "topic") for e in entities)


def test_graph_builder_get_relationship_list() -> None:
    b = GraphBuilder("hh1")
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Call Mom", "confidence": 0.9}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "intent", "payload": {"name": "call", "slots": {"contact": "Mom"}, "confidence": 0.85}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:02:00Z", "seq": 2, "event_type": "intent", "payload": {"name": "reminder", "slots": {"contact": "Doctor"}, "confidence": 0.8}},
    ]
    b.process_events(events, "s1", "d1")
    rels = b.get_relationship_list()
    assert isinstance(rels, list)
    for r in rels:
        assert "src_entity_id" in r
        assert "dst_entity_id" in r
        assert r.get("rel_type") in ("CO_OCCURS", "TRIGGERED")


def test_build_hetero_from_tables_empty_entities() -> None:
    sessions = [{"id": "s1"}]
    utterances = [{"id": "u1", "session_id": "s1", "intent": "call"}]
    entities = []
    mentions = []
    relationships = []
    data = build_hetero_from_tables("hh1", sessions, utterances, entities, mentions, relationships, devices=[])
    assert data["entity"].num_nodes == 0
    assert data["person"].num_nodes == 1


def test_build_hetero_from_tables_no_devices() -> None:
    sessions = [{"id": "s1"}]
    utterances = [{"id": "u1", "session_id": "s1"}]
    entities = [{"id": "e1"}]
    mentions = [{"utterance_id": "u1", "entity_id": "e1"}]
    relationships = []
    data = build_hetero_from_tables("hh1", sessions, utterances, entities, mentions, relationships, devices=None)
    assert data["device"].num_nodes == 0
