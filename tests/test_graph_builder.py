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


def test_graph_builder_financial_events_normalized() -> None:
    """Financial event types (transaction_detected, payee_added, bank_alert_received) normalize into entities/mentions like other events."""
    builder = GraphBuilder("hh1")
    session_id = "s1"
    device_id = "d1"
    events = [
        {"id": "e1", "session_id": session_id, "device_id": device_id, "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "transaction_detected", "payload": {"merchant": "Acme Corp", "confidence": 0.9}},
        {"id": "e2", "session_id": session_id, "device_id": device_id, "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "payee_added", "payload": {"payee_name": "Jane", "payee_type": "person", "confidence": 0.85}},
        {"id": "e3", "session_id": session_id, "device_id": device_id, "ts": "2024-01-01T10:02:00Z", "seq": 2, "event_type": "bank_alert_received", "payload": {"account_id_hash": "acc_abc123", "confidence": 0.8}},
    ]
    builder.process_events(events, session_id, device_id)
    entities = builder.get_entity_list()
    entity_types = {e["entity_type"] for e in entities}
    assert "merchant" in entity_types
    assert "person" in entity_types
    assert "account" in entity_types
    mentions = builder.get_mention_list()
    assert len(mentions) >= 3


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
    # Event-centric: one event per utterance, session->event, event->entity (mentions), event->next_event
    assert data["event"].num_nodes == 2
    assert data["session", "has_event", "event"].edge_index.size(1) == 2
    assert data["event", "mentions", "entity"].edge_index.size(1) == 2
    assert data["event", "next_event", "event"].edge_index.size(1) == 1  # u1 -> u2 in s1
    if ("entity", "co_occurs", "entity") in data.edge_stores:
        assert data["entity", "co_occurs", "entity"].edge_index.size(1) == 1
