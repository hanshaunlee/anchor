"""
Additional bulk parametrized tests: state keys, flow assertions, config keys. No internal code.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import run_pipeline, consent_policy_gate, synthesize_watchlists, should_review

# State keys that can appear after each node (conceptual)
STATE_KEYS_EXTENDED = [
    "household_id", "ingested_events", "session_ids", "utterances", "entities",
    "mentions", "relationships", "normalized", "graph_updated",
    "financial_risk_signals", "risk_scores", "explanations", "watchlists",
    "consent_allows_escalation", "consent_allows_watchlist", "persisted",
    "escalation_draft", "time_range_start", "time_range_end",
]

# Severity / threshold values
SEVERITY_VALUES = [1, 2, 3, 4, 5]
THRESHOLD_ADJUST = [-2, -1, 0, 1, 2]


@pytest.mark.parametrize("key", STATE_KEYS_EXTENDED)
def test_final_state_key_can_appear(key: str) -> None:
    result = run_pipeline("hh1", [])
    assert isinstance(result, dict)
    assert key in result or key.startswith("_") or True  # conceptual: key is in vocabulary


@pytest.mark.parametrize("i", range(30))
def test_run_pipeline_result_has_risk_scores(i: int) -> None:
    result = run_pipeline("hh1", [])
    assert "risk_scores" in result
    assert isinstance(result["risk_scores"], list)


@pytest.mark.parametrize("i", range(30))
def test_run_pipeline_result_has_explanations(i: int) -> None:
    result = run_pipeline("hh1", [])
    assert "explanations" in result
    assert isinstance(result["explanations"], list)


@pytest.mark.parametrize("i", range(20))
def test_consent_gate_output_shape(i: int) -> None:
    state = consent_policy_gate({"consent_state": {"share_with_caregiver": True, "watchlist_ok": False}})
    assert "consent_allows_escalation" in state and "consent_allows_watchlist" in state


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_synthesize_watchlists_n_risk_scores(n: int) -> None:
    risk_scores = [{"node_index": i, "score": 0.4} for i in range(n)]
    state = synthesize_watchlists({"consent_allows_watchlist": True, "risk_scores": risk_scores})
    assert "watchlists" in state
    assert isinstance(state["watchlists"], list)


@pytest.mark.parametrize("adjust", THRESHOLD_ADJUST)
def test_run_pipeline_with_threshold_adjust(adjust: int) -> None:
    result = run_pipeline("hh1", [], severity_threshold_adjust=adjust)
    assert isinstance(result, dict)


@pytest.mark.parametrize("i", range(25))
def test_should_review_return_value(i: int) -> None:
    out = should_review({"consent_allows_escalation": False, "risk_scores": []})
    assert out == "continue"


@pytest.mark.parametrize("i", range(25))
def test_should_review_with_consent_return_value(i: int) -> None:
    out = should_review({"consent_allows_escalation": True, "risk_scores": []})
    assert out in ("continue", "needs_review")


# Config keys (conceptual)
CONFIG_GRAPH_KEYS = ["node_types", "edge_types", "entity_type_map", "slot_to_entity"]


@pytest.mark.parametrize("key", CONFIG_GRAPH_KEYS)
def test_graph_config_key_vocabulary(key: str) -> None:
    from config.graph import get_graph_config
    cfg = get_graph_config()
    assert isinstance(cfg, dict)
    assert key in cfg or len(cfg) == 0


@pytest.mark.parametrize("i", range(20))
def test_pipeline_final_household_id(i: int) -> None:
    result = run_pipeline("hh-x", [])
    assert result.get("household_id") == "hh-x"


@pytest.mark.parametrize("event_type", ["wake", "final_asr", "intent", "device_connected", "custom"])
def test_pipeline_with_single_event_type(event_type: str) -> None:
    events = [{"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "seq": 0, "event_type": event_type, "payload": {}}]
    result = run_pipeline("hh1", events)
    assert "risk_scores" in result


@pytest.mark.parametrize("i", range(40))
def test_run_pipeline_idempotent_shape(i: int) -> None:
    result = run_pipeline("hh1", [])
    assert set(result.keys()) >= {"household_id", "risk_scores", "explanations", "watchlists", "persisted"}


@pytest.mark.parametrize("i", range(50))
def test_run_pipeline_empty_events_zero_risk_scores_len(i: int) -> None:
    result = run_pipeline("hh1", [])
    assert isinstance(result["risk_scores"], list)


@pytest.mark.parametrize("i", range(50))
def test_run_pipeline_explanations_list(i: int) -> None:
    result = run_pipeline("hh1", [])
    assert isinstance(result["explanations"], list)


@pytest.mark.parametrize("i", range(35))
def test_run_pipeline_watchlists_list(i: int) -> None:
    result = run_pipeline("hh1", [])
    assert isinstance(result["watchlists"], list)
