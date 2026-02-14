"""Tests for config.settings: PipelineSettings, MLSettings, get_pipeline_settings, get_ml_settings."""
import os
from unittest.mock import patch

import pytest

# Clear lru_cache so env overrides take effect in tests
@pytest.fixture(autouse=True)
def clear_settings_cache():
    try:
        from config.settings import get_pipeline_settings, get_ml_settings
        get_pipeline_settings.cache_clear()
        get_ml_settings.cache_clear()
    except ImportError:
        pass
    yield
    try:
        get_pipeline_settings.cache_clear()
        get_ml_settings.cache_clear()
    except Exception:
        pass


def test_pipeline_settings_defaults() -> None:
    from config.settings import PipelineSettings
    s = PipelineSettings()
    assert s.risk_score_threshold == 0.5
    assert s.explanation_score_min == 0.4
    assert s.watchlist_score_min == 0.5
    assert s.escalation_score_min == 0.6
    assert s.severity_threshold >= 1 and s.severity_threshold <= 5
    assert s.consent_share_key == "share_with_caregiver"
    assert s.consent_watchlist_key == "watchlist_ok"
    assert s.default_consent_share is True
    assert s.default_consent_watchlist is True
    assert 1 <= s.timeline_snippet_max <= 50


def test_ml_settings_defaults() -> None:
    from config.settings import MLSettings
    s = MLSettings()
    assert s.embedding_dim >= 8 and s.embedding_dim <= 512
    assert s.risk_inference_entity_cap >= 1
    assert 0.01 <= s.calibration_adjust_step <= 1.0
    assert s.model_version_tag == "v0"


def test_get_pipeline_settings_cached() -> None:
    from config.settings import get_pipeline_settings
    a = get_pipeline_settings()
    b = get_pipeline_settings()
    assert a is b


def test_get_ml_settings_cached() -> None:
    from config.settings import get_ml_settings
    a = get_ml_settings()
    b = get_ml_settings()
    assert a is b


@patch.dict(os.environ, {"ANCHOR_risk_score_threshold": "0.7", "ANCHOR_severity_threshold": "3"}, clear=False)
def test_pipeline_settings_env_override() -> None:
    from config.settings import get_pipeline_settings
    get_pipeline_settings.cache_clear()
    s = get_pipeline_settings()
    assert s.risk_score_threshold == 0.7
    assert s.severity_threshold == 3


def test_get_ml_config_path_default() -> None:
    from config.settings import get_ml_config_path
    with patch.dict(os.environ, {}, clear=True):
        if "ANCHOR_ML_CONFIG" in os.environ:
            del os.environ["ANCHOR_ML_CONFIG"]
    path = get_ml_config_path()
    assert path == "configs/hgt_baseline.yaml" or "hgt_baseline" in path
