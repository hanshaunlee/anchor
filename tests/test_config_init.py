"""Tests for config package __init__: exports and getters."""
import pytest


def test_config_exports() -> None:
    """config package exports get_pipeline_settings, get_ml_settings, get_graph_config."""
    import config
    assert hasattr(config, "get_pipeline_settings")
    assert hasattr(config, "get_ml_settings")
    assert hasattr(config, "get_graph_config")
    assert config.get_pipeline_settings is not None
    assert config.get_ml_settings is not None
    assert config.get_graph_config is not None


def test_config_get_pipeline_settings_returns_settings() -> None:
    """get_pipeline_settings() returns an object with risk_score_threshold, etc."""
    from config import get_pipeline_settings
    s = get_pipeline_settings()
    assert hasattr(s, "risk_score_threshold")
    assert hasattr(s, "watchlist_score_min")
    assert hasattr(s, "severity_threshold")
    assert hasattr(s, "consent_share_key")


def test_config_get_ml_settings_returns_settings() -> None:
    """get_ml_settings() returns an object with calibration_adjust_step, etc."""
    from config import get_ml_settings
    s = get_ml_settings()
    assert hasattr(s, "calibration_adjust_step")


def test_config_get_graph_config_returns_dict() -> None:
    """get_graph_config() returns dict with entity_type_map, slot_to_entity, etc."""
    from config import get_graph_config
    g = get_graph_config()
    assert isinstance(g, dict)
    assert "entity_type_map" in g or "slot_to_entity" in g or "event_types" in g
