"""Tests for pipeline financial_security_agent node and exception path."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import financial_security_agent


def test_financial_security_agent_node_empty_events() -> None:
    state = {
        "household_id": "hh1",
        "ingested_events": [],
        "utterances": [],
        "entities": [],
        "mentions": [],
        "relationships": [],
        "consent_state": {},
    }
    out = financial_security_agent(state)
    assert "financial_risk_signals" in out
    assert "financial_watchlists" in out
    assert "financial_logs" in out
    assert isinstance(out["financial_risk_signals"], list)
    assert isinstance(out["financial_watchlists"], list)


def test_financial_security_agent_node_with_events() -> None:
    state = {
        "household_id": "hh1",
        "ingested_events": [
            {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Medicare called", "confidence": 0.9}},
            {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "intent", "payload": {"name": "share_ssn", "slots": {}, "confidence": 0.8}},
        ],
        "utterances": [],
        "entities": [],
        "mentions": [],
        "relationships": [],
        "consent_state": {"share_with_caregiver": True, "watchlist_ok": True},
    }
    out = financial_security_agent(state)
    assert "financial_risk_signals" in out
    assert "financial_logs" in out
    assert "logs" in out
    assert any("financial" in str(m).lower() for m in out.get("logs", []))


def test_financial_security_agent_node_consent_in_state() -> None:
    state = {
        "household_id": "hh1",
        "ingested_events": [],
        "consent_state": {"share_with_caregiver": False, "watchlist_ok": True},
    }
    out = financial_security_agent(state)
    assert "financial_watchlists" in out
