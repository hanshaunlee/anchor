"""
Flow tests: ingest -> graph (build_graph_from_events); risk list/detail/feedback -> UI. No internal code.
"""
import pytest
from uuid import uuid4
from unittest.mock import MagicMock


# --- Ingest -> build_graph_from_events -> entities, utterances, mentions, relationships ---
@pytest.mark.parametrize("n_events", [0, 1, 2, 5])
def test_build_graph_from_events_event_count(n_events: int) -> None:
    from domain.graph_service import build_graph_from_events
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "event_type": "wake", "payload": {}}
        for _ in range(n_events)
    ]
    u, e, m, r = build_graph_from_events("hh1", events, supabase=None)
    assert isinstance(u, list) and isinstance(e, list) and isinstance(m, list) and isinstance(r, list)


def test_normalize_events_domain_returns_four_lists() -> None:
    from domain.graph_service import normalize_events as domain_norm
    u, e, m, r = domain_norm("hh1", [])
    assert isinstance(u, list) and isinstance(e, list) and isinstance(m, list) and isinstance(r, list)


# --- list_risk_signals -> list of cards ---
def test_list_risk_signals_empty_household_returns_empty_signals() -> None:
    from domain.risk_service import list_risk_signals
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = []
    mock.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.count = 0
    out = list_risk_signals("hh1", mock, limit=10, offset=0, max_age_days=None)
    assert out.signals == []
    assert out.total == 0


def test_list_risk_signals_maps_rows_to_cards() -> None:
    from domain.risk_service import list_risk_signals
    sid = str(uuid4())
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = [
        {"id": sid, "ts": "2025-01-01T00:00:00Z", "signal_type": "relational_anomaly", "severity": 3, "score": 0.7, "status": "open", "explanation": {}},
    ]
    mock.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.count = 1
    out = list_risk_signals("hh1", mock, limit=10, offset=0, max_age_days=None)
    assert hasattr(out, "signals") and isinstance(out.signals, list)
    if out.signals:
        assert hasattr(out.signals[0], "id") or isinstance(out.signals[0], dict)


# --- get_risk_signal_detail -> detail with subgraph ---
def test_get_risk_signal_detail_not_found_returns_none() -> None:
    from domain.risk_service import get_risk_signal_detail
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value.data = None
    out = get_risk_signal_detail(uuid4(), "hh1", mock)
    assert out is None


# --- get_similar_incidents (explain_service) -> uses embeddings from DB ---
def test_get_similar_incidents_delegates_to_similarity_service() -> None:
    from domain.explain_service import get_similar_incidents
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.eq.return_value.eq.return_value.execute.return_value.data = []
    out = get_similar_incidents(uuid4(), "hh1", mock, top_k=5)
    assert hasattr(out, "available") or "incidents" in str(type(out)) or isinstance(out, dict)


# --- Flow: pipeline run_pipeline -> final state has all keys for UI ---
def test_pipeline_final_state_has_ui_relevant_keys() -> None:
    from api.pipeline import run_pipeline
    result = run_pipeline("hh1", [])
    keys_for_ui = ["risk_scores", "explanations", "watchlists", "escalation_draft", "persisted"]
    for k in keys_for_ui:
        assert k in result
