"""Unit tests for graph builder."""
import pytest

pytest.importorskip("torch")
from ml.graph.builder import GraphBuilder, build_hetero_from_tables


def test_graph_builder_utterances_and_entities() -> None:
    builder = GraphBuilder("hh1")
    session_id = "s1"
    device_id = "d1"
    events = [
        {"id": "e1", "session_id": session_id, "device_id": device_id, "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
        {"id": "e2", "session_id": session_id, "device_id": device_id, "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "Call John", "confidence": 0.9, "speaker": {"role": "elder"}}},
        {"id": "e3", "session_id": session_id, "device_id": device_id, "ts": "2024-01-01T10:01:01Z", "seq": 2, "event_type": "intent", "payload": {"name": "call", "slots": {"contact": "John"}, "confidence": 0.85}},
    ]
    builder.process_events(events, session_id, device_id)
    utterances = builder.get_utterance_list()
    assert len(utterances) == 1
    assert utterances[0]["session_id"] == session_id
    assert utterances[0].get("intent") == "call"
    entities = builder.get_entity_list()
    assert len(entities) >= 1
    mentions = builder.get_mention_list()
    assert len(mentions) >= 1


def test_build_hetero_from_tables() -> None:
    sessions = [{"id": "s1"}, {"id": "s2"}]
    utterances = [
        {"id": "u1", "session_id": "s1", "intent": "call"},
        {"id": "u2", "session_id": "s1", "intent": "reminder"},
    ]
    entities = [{"id": "e1"}, {"id": "e2"}]
    mentions = [{"utterance_id": "u1", "entity_id": "e1"}, {"utterance_id": "u2", "entity_id": "e2"}]
    relationships = [
        {"src_entity_id": "e1", "dst_entity_id": "e2", "rel_type": "CO_OCCURS", "first_seen_at": 0, "last_seen_at": 100, "count": 1, "weight": 1.0},
    ]
    data = build_hetero_from_tables("hh1", sessions, utterances, entities, mentions, relationships, devices=[{"id": "d1"}])
    assert data["person"].num_nodes == 1
    assert data["entity"].num_nodes == 2
    if ("entity", "co_occurs", "entity") in data.edge_stores:
        assert data["entity", "co_occurs", "entity"].edge_index.size(1) == 1
