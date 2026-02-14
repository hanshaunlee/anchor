"""Extended tests for worker.jobs: run_risk_inference cap, run_pipeline with financial outputs."""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from worker.jobs import run_risk_inference, run_pipeline, _ml_settings, _pipeline_settings


def test_run_risk_inference_respects_cap() -> None:
    from config.settings import get_ml_settings
    cap = get_ml_settings().risk_inference_entity_cap
    entities = [{"id": f"e{i}"} for i in range(cap + 50)]
    graph_data = {"entities": entities, "mentions": [], "relationships": [], "utterances": []}
    scores = run_risk_inference("hh1", graph_data)
    assert len(scores) <= cap


def test_run_risk_inference_signal_shape() -> None:
    graph_data = {
        "entities": [{"id": "e1"}],
        "mentions": [],
        "relationships": [],
        "utterances": [],
    }
    scores = run_risk_inference("hh1", graph_data)
    assert len(scores) == 1
    s = scores[0]
    assert "household_id" in s
    assert "signal_type" in s
    assert "severity" in s
    assert "score" in s
    assert "explanation" in s
    assert "status" in s


def test_run_pipeline_contains_financial_outputs() -> None:
    result = run_pipeline(None, "hh1", None, None)
    assert "financial_risk_signals" in result or "risk_scores" in result
    assert "financial_watchlists" in result or "watchlists" in result
    assert "normalized" in result
    assert result.get("normalized") is True


def test_ml_settings_fallback_has_attrs() -> None:
    s = _ml_settings()
    assert hasattr(s, "embedding_dim")
    assert hasattr(s, "risk_inference_entity_cap")
    assert hasattr(s, "model_version_tag")


def test_pipeline_settings_fallback_has_persist_score_min() -> None:
    s = _pipeline_settings()
    assert hasattr(s, "persist_score_min")
