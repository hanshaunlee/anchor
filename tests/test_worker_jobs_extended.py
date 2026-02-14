"""Extended tests for worker.jobs: run_risk_inference cap, run_pipeline, centroid watchlist match."""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from worker.jobs import (
    run_risk_inference,
    run_pipeline,
    _ml_settings,
    _pipeline_settings,
    _check_embedding_centroid_watchlists,
)


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


def test_check_embedding_centroid_watchlists_creates_match_signal_on_match() -> None:
    """When embedding matches a centroid watchlist, a watchlist_embedding_match risk_signal is created."""
    household_id = "hh-centroid"
    risk_signal_id = "rs-orig"
    vec = [0.5] * 8  # 8-dim so we can use a matching centroid
    centroid = [0.5] * 8
    threshold = 0.8

    tables = {}
    def table(name):
        if name not in tables:
            t = MagicMock()
            t.select.return_value = t
            t.eq.return_value = t
            t.single.return_value = t
            if name == "watchlists":
                t.execute.return_value.data = [
                    {"id": "wl-1", "pattern": {"centroid": centroid, "threshold": threshold, "model_name": "hgt"}, "expires_at": None},
                ]
            elif name == "risk_signals":
                t.execute.return_value.data = {"explanation": {}}
                t.update.return_value = t
                t.insert.return_value.execute.return_value.data = [{"id": "rs-match"}]
            tables[name] = t
        return tables[name]

    mock_sb = MagicMock()
    mock_sb.table.side_effect = table

    _check_embedding_centroid_watchlists(mock_sb, household_id, risk_signal_id, vec)

    # Must have inserted a watchlist_embedding_match risk_signal
    risk_signals_table = tables["risk_signals"]
    risk_signals_table.insert.assert_called_once()
    call_args = risk_signals_table.insert.call_args[0][0]
    assert call_args.get("signal_type") == "watchlist_embedding_match"
    assert call_args.get("household_id") == household_id
    assert 3 <= call_args.get("severity", 0) <= 5
    assert "watchlist_match" in (call_args.get("explanation") or {})
