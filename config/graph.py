"""
Graph schema and builder constants. Single source of truth for node types, edge types,
entity types, event types, and feature dimensions. Overridable via YAML for scalability.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

# Default schema (used when no YAML or for fallback)
DEFAULT_NODE_TYPES = [
    "person",
    "device",
    "session",
    "utterance",
    "intent",
    "entity",
]
DEFAULT_EDGE_TYPES = [
    ("person", "uses", "device"),
    ("session", "has", "utterance"),
    ("utterance", "expresses", "intent"),
    ("utterance", "mentions", "entity"),
    ("entity", "co_occurs", "entity"),
]
DEFAULT_ENTITY_TYPES: dict[str, str] = {
    "phone": "phone",
    "email": "email",
    "person": "person",
    "org": "org",
    "merchant": "merchant",
    "topic": "topic",
    "account": "account",
    "device": "device",
    "location": "location",
}
# Slot name substring -> entity type (for intent slots)
DEFAULT_SLOT_TO_ENTITY: dict[str, str] = {
    "phone": "phone",
    "number": "phone",
    "email": "email",
    "name": "person",
    "person": "person",
    "merchant": "merchant",
}
DEFAULT_EVENT_TYPES = frozenset({"final_asr", "intent", "watchlist_hit"})
DEFAULT_SPEAKER_ROLES = frozenset({"elder", "agent", "unknown"})
DEFAULT_BASE_FEATURE_DIMS: dict[str, int] = {
    "person": 8,
    "device": 8,
    "session": 8,
    "utterance": 16,
    "intent": 8,
    "entity": 16,
}
DEFAULT_EDGE_ATTR_DIM = 4
DEFAULT_TIME_ENCODING_DIM = 8
DEFAULT_CO_OCCURRENCE_WINDOW_SEC = 300
DEFAULT_PERSON_IDS = ["person_elder"]
# Explainer motif keywords (can be overridden by config)
DEFAULT_URGENCY_TOPICS: frozenset[str] = frozenset({
    "medicare", "irs", "social security", "ssn", "urgent", "immediately",
    "suspended", "account", "verify", "confirm",
})
DEFAULT_SENSITIVE_INTENTS: frozenset[str] = frozenset({
    "sensitive_request", "share_ssn", "share_card", "pay_now", "wire_money", "buy_gift_cards",
})


def _load_graph_yaml() -> dict[str, Any]:
    path = os.environ.get("ANCHOR_GRAPH_CONFIG", "")
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_graph_config() -> dict[str, Any]:
    """
    Graph configuration: node_types, edge_types, entity_type_map, slot_to_entity,
    event_types, speaker_roles, base_feature_dims, time_encoding_dim, co_occurrence_window_sec, etc.
    Merges YAML (if ANCHOR_GRAPH_CONFIG set) with defaults.
    """
    data = _load_graph_yaml()
    return {
        "node_types": data.get("node_types", DEFAULT_NODE_TYPES),
        "edge_types": data.get("edge_types", DEFAULT_EDGE_TYPES),
        "entity_type_map": {**DEFAULT_ENTITY_TYPES, **data.get("entity_type_map", {})},
        "slot_to_entity": {**DEFAULT_SLOT_TO_ENTITY, **data.get("slot_to_entity", {})},
        "event_types": frozenset(data.get("event_types", list(DEFAULT_EVENT_TYPES))),
        "speaker_roles": frozenset(data.get("speaker_roles", list(DEFAULT_SPEAKER_ROLES))),
        "base_feature_dims": {**DEFAULT_BASE_FEATURE_DIMS, **data.get("base_feature_dims", {})},
        "edge_attr_dim": data.get("edge_attr_dim", DEFAULT_EDGE_ATTR_DIM),
        "time_encoding_dim": data.get("time_encoding_dim", DEFAULT_TIME_ENCODING_DIM),
        "co_occurrence_window_sec": data.get("co_occurrence_window_sec", DEFAULT_CO_OCCURRENCE_WINDOW_SEC),
        "person_ids": data.get("person_ids", DEFAULT_PERSON_IDS),
        "urgency_topics": frozenset(data.get("urgency_topics", list(DEFAULT_URGENCY_TOPICS))),
        "sensitive_intents": frozenset(data.get("sensitive_intents", list(DEFAULT_SENSITIVE_INTENTS))),
    }
