"""Tests for config.graph: get_graph_config, defaults."""
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_graph_config_cache():
    try:
        from config.graph import get_graph_config
        get_graph_config.cache_clear()
    except ImportError:
        pass
    yield
    try:
        get_graph_config.cache_clear()
    except Exception:
        pass


def test_get_graph_config_returns_dict() -> None:
    from config.graph import get_graph_config
    g = get_graph_config()
    assert isinstance(g, dict)


def test_graph_config_has_node_types() -> None:
    from config.graph import get_graph_config, DEFAULT_NODE_TYPES
    g = get_graph_config()
    assert "node_types" in g
    assert g["node_types"] == DEFAULT_NODE_TYPES or isinstance(g["node_types"], list)
    assert "person" in g["node_types"]
    assert "entity" in g["node_types"]
    assert "event" in g["node_types"]


def test_graph_config_has_edge_types() -> None:
    from config.graph import get_graph_config, DEFAULT_EDGE_TYPES
    g = get_graph_config()
    assert "edge_types" in g
    assert ("entity", "co_occurs", "entity") in g["edge_types"] or any(
        "co_occurs" in str(e) for e in g["edge_types"]
    )
    # Event-centric: session->event, event->entity, event->next_event
    edge_tuples = [tuple(e) if isinstance(e, list) else e for e in g["edge_types"]]
    assert ("session", "has_event", "event") in edge_tuples
    assert ("event", "mentions", "entity") in edge_tuples
    assert ("event", "next_event", "event") in edge_tuples


def test_graph_config_entity_type_map() -> None:
    from config.graph import get_graph_config, DEFAULT_ENTITY_TYPES
    g = get_graph_config()
    assert "entity_type_map" in g
    for k, v in DEFAULT_ENTITY_TYPES.items():
        assert g["entity_type_map"].get(k) == v


def test_graph_config_slot_to_entity() -> None:
    from config.graph import get_graph_config
    g = get_graph_config()
    assert "slot_to_entity" in g
    assert g["slot_to_entity"].get("phone") == "phone"
    assert g["slot_to_entity"].get("name") == "person"


def test_graph_config_event_types_frozenset() -> None:
    from config.graph import get_graph_config
    g = get_graph_config()
    assert "event_types" in g
    assert isinstance(g["event_types"], frozenset)
    assert "final_asr" in g["event_types"]
    assert "intent" in g["event_types"]


def test_graph_config_urgency_and_sensitive() -> None:
    from config.graph import get_graph_config
    g = get_graph_config()
    assert "urgency_topics" in g
    assert "sensitive_intents" in g
    assert isinstance(g["urgency_topics"], frozenset)
    assert isinstance(g["sensitive_intents"], frozenset)
