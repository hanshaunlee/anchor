"""Tests for api.graph_state: AnchorState, append_log."""
import pytest

from api.graph_state import AnchorState, append_log


def test_anchor_state_defaults() -> None:
    state = AnchorState()
    assert state.household_id == ""
    assert state.ingested_events == []
    assert state.normalized is False
    assert state.graph_updated is False
    assert state.risk_scores == []
    assert state.consent_allows_escalation is True
    assert state.consent_allows_watchlist is True
    assert state.persisted is False
    assert state.logs == []


def test_anchor_state_from_dict() -> None:
    state = AnchorState(
        household_id="hh1",
        ingested_events=[{"event_type": "wake"}],
        normalized=True,
    )
    assert state.household_id == "hh1"
    assert len(state.ingested_events) == 1
    assert state.normalized is True


def test_append_log_creates_list() -> None:
    state = {}
    append_log(state, "first")
    assert state["logs"] == ["first"]


def test_append_log_appends() -> None:
    state = {"logs": ["a"]}
    append_log(state, "b")
    assert state["logs"] == ["a", "b"]


def test_append_log_returns_state() -> None:
    state = {}
    out = append_log(state, "msg")
    assert out is state
    assert out["logs"] == ["msg"]
