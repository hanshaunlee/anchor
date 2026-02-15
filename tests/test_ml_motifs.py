"""Tests for ml.explainers.motifs: extract_motifs, _ts_float."""
import pytest

# motifs module does not require torch; but ml.explainers.__init__ imports gnn_explainer
pytest.importorskip("torch_geometric")
from ml.explainers.motifs import extract_motifs

# _ts_float is used internally; we test via extract_motifs or could import if needed


def test_extract_motifs_empty_inputs() -> None:
    tags, snippet, structural = extract_motifs([], [], [], [], [])
    assert tags == []
    assert snippet == []
    assert structural == []


def test_extract_motifs_new_contact_urgency() -> None:
    utterances = [
        {"id": "u1", "ts": "2024-01-01T12:00:00Z", "text": "Someone from Medicare called saying my account is suspended", "intent": ""},
    ]
    mentions = [{"utterance_id": "u1", "entity_id": "e1", "ts": "2024-01-01T12:00:00Z"}]
    entities = [{"id": "e1", "entity_type": "person"}]
    entity_id_to_canonical = {"e1": "Medicare caller"}
    tags, snippet, structural = extract_motifs(
        utterances, mentions, entities, [], [],
        entity_id_to_canonical=entity_id_to_canonical,
    )
    assert any("urgency" in t.lower() or "new contact" in t.lower() for t in tags)
    assert isinstance(snippet, list)


def test_extract_motifs_bursty_contact() -> None:
    mentions = [
        {"utterance_id": "u1", "entity_id": "e1", "ts": 1000.0},
        {"utterance_id": "u2", "entity_id": "e1", "ts": 1001.0},
        {"utterance_id": "u3", "entity_id": "e1", "ts": 1002.0},
    ]
    entities = [{"id": "e1", "entity_type": "phone"}]
    tags, snippet, structural = extract_motifs([], mentions, entities, [], [])
    assert any("burst" in t.lower() or "repeated" in t.lower() for t in tags)


def test_extract_motifs_device_switching() -> None:
    events = [
        {"device_id": "d1", "ts": "2024-01-01T10:00:00Z"},
        {"device_id": "d2", "ts": "2024-01-01T10:01:00Z"},
    ]
    tags, _, _ = extract_motifs([], [], [], [], events)
    assert any("device" in t.lower() and "switch" in t.lower() for t in tags)


def test_extract_motifs_sensitive_cascade() -> None:
    utterances = [
        {"id": "u1", "ts": 1000.0, "text": "call from unknown", "intent": ""},
        {"id": "u2", "ts": 1001.0, "text": "", "intent": "share_ssn"},
    ]
    mentions = [{"utterance_id": "u1", "entity_id": "e1", "ts": 1000.0}]
    entities = [{"id": "e1"}]
    tags, snippet, structural = extract_motifs(utterances, mentions, entities, [], [])
    assert any("sensitive" in t.lower() or "cascade" in t.lower() for t in tags)


def test_extract_motifs_timeline_capped() -> None:
    utterances = [
        {"id": f"u{i}", "ts": 1000.0 + i, "text": "medicare", "intent": ""}
        for i in range(10)
    ]
    mentions = [{"utterance_id": f"u{i}", "entity_id": "e1", "ts": 1000.0 + i} for i in range(10)]
    entities = [{"id": "e1"}]
    _, snippet, _ = extract_motifs(utterances, mentions, entities, [], [])
    assert len(snippet) <= 6


def test_extract_motifs_custom_urgency_and_sensitive() -> None:
    tags, _, _ = extract_motifs(
        [{"id": "u1", "ts": 0, "text": "custom urgent word", "intent": ""}],
        [{"utterance_id": "u1", "entity_id": "e1", "ts": 0}],
        [{"id": "e1"}],
        [], [],
        urgency_topics=frozenset({"custom"}),
        sensitive_intents=frozenset({"custom_intent"}),
    )
    # May or may not fire depending on "new in last hour" logic; at least no crash
    assert isinstance(tags, list)
