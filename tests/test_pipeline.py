"""End-to-end pipeline run on synthetic data."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")
from api.pipeline import run_pipeline


def test_pipeline_synthetic() -> None:
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "Remind me", "confidence": 0.9}},
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:01Z", "seq": 2, "event_type": "intent", "payload": {"name": "reminder", "slots": {}, "confidence": 0.85}},
    ]
    result = run_pipeline("hh1", events, "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
    assert result is not None
    assert result.get("normalized") is True
    assert result.get("graph_updated") is True
    assert "risk_scores" in result
    assert "watchlists" in result
    assert result.get("persisted") is True
    assert "logs" in result
    # Financial Security Agent node runs after graph_update
    assert "financial_logs" in result
    assert isinstance(result.get("financial_risk_signals"), list)
    assert isinstance(result.get("financial_watchlists"), list)
