"""
Specification-based graph builder tests: exact slot->entity mapping, utterance text from payload.
"""
import pytest

pytest.importorskip("torch")
from ml.graph.builder import GraphBuilder


def test_utterance_text_from_final_asr_payload() -> None:
    """Utterance text must come from the final_asr event payload."""
    builder = GraphBuilder("hh1")
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "Call Mom at 5pm", "confidence": 0.9}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:01Z", "seq": 2, "event_type": "intent", "payload": {"name": "reminder", "slots": {"time": "5pm"}, "confidence": 0.85}},
    ]
    builder.process_events(events, "s1", "d1")
    utterances = builder.get_utterance_list()
    assert len(utterances) == 1
    assert utterances[0]["text"] == "Call Mom at 5pm"
    assert utterances[0]["intent"] == "reminder"


def test_slot_name_maps_to_person_entity_type() -> None:
    """Config slot_to_entity: 'name' -> person. Intent slot 'name': 'Alice' must create person entity."""
    builder = GraphBuilder("hh1")
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "intent", "payload": {"name": "call", "slots": {"name": "Alice"}, "confidence": 0.9}},
    ]
    builder.process_events(events, "s1", "d1")
    entities = builder.get_entity_list()
    assert len(entities) >= 1
    person_entities = [e for e in entities if e.get("entity_type") == "person"]
    assert len(person_entities) == 1
    assert person_entities[0].get("canonical") == "Alice"


def test_co_occurrence_creates_relationship() -> None:
    """Two entities in same session window must get CO_OCCURS relationship."""
    builder = GraphBuilder("hh1")
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "intent", "payload": {"name": "call", "slots": {"name": "A"}, "confidence": 0.9}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:01Z", "seq": 1, "event_type": "intent", "payload": {"name": "reminder", "slots": {"name": "B"}, "confidence": 0.9}},
    ]
    builder.process_events(events, "s1", "d1")
    rels = builder.get_relationship_list()
    co_occurs = [r for r in rels if r.get("rel_type") == "CO_OCCURS"]
    assert len(co_occurs) >= 1
    assert co_occurs[0]["rel_type"] == "CO_OCCURS"
    assert "src_entity_id" in co_occurs[0] and "dst_entity_id" in co_occurs[0]
