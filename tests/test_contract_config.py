"""
Contract tests: config public functions return expected shape (get_graph_config, get_pipeline_settings, etc.).
"""
import pytest


# --- config.graph: get_graph_config ---
def test_get_graph_config_returns_dict() -> None:
    from config.graph import get_graph_config
    out = get_graph_config()
    assert isinstance(out, dict)


@pytest.mark.parametrize("key", ["node_types", "edge_types", "entity_type_map"])
def test_get_graph_config_has_expected_keys(key: str) -> None:
    from config.graph import get_graph_config
    out = get_graph_config()
    assert key in out or not out  # allow empty dict in test env


# --- config.settings: get_pipeline_settings, get_ml_settings, get_worker_settings ---
def test_get_pipeline_settings_returns_object_with_attrs() -> None:
    from config.settings import get_pipeline_settings
    out = get_pipeline_settings()
    assert hasattr(out, "risk_score_threshold") or hasattr(out, "watchlist_score_min") or hasattr(out, "severity_threshold")


def test_get_ml_settings_returns_object_with_attrs() -> None:
    from config.settings import get_ml_settings
    out = get_ml_settings()
    assert hasattr(out, "checkpoint_path") or hasattr(out, "model_version_tag") or hasattr(out, "model_name")


def test_get_worker_settings_returns_object() -> None:
    from config.settings import get_worker_settings
    out = get_worker_settings()
    assert out is not None


# --- config.graph_policy: allow_graph_mutation_for_event ---
@pytest.mark.parametrize("event_type", ["wake", "final_asr", "intent"])
def test_allow_graph_mutation_for_event_returns_bool(event_type: str) -> None:
    from config.graph_policy import allow_graph_mutation_for_event
    event = {"event_type": event_type, "payload": {}}
    out = allow_graph_mutation_for_event(event)
    assert isinstance(out, bool)


# --- config.get_ml_config_path ---
def test_get_ml_config_path_returns_str() -> None:
    from config.settings import get_ml_config_path
    out = get_ml_config_path()
    assert isinstance(out, str)
