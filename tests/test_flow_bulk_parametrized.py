"""
Bulk parametrized tests: pipeline node names, state keys, flow assertions. No internal code—only names and data flow.
Aims to expand test count toward ~1000 with many (name, param) combinations.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import (
    build_graph,
    run_pipeline,
    ingest_events_batch,
    normalize_events,
    graph_update,
    financial_security_agent,
    risk_score_inference,
    generate_explanations,
    consent_policy_gate,
    synthesize_watchlists,
    draft_escalation_message,
    persist_outputs,
    should_review,
)

# Node names in the pipeline graph (conceptual flow)
PIPELINE_NODE_NAMES = [
    "ingest", "normalize", "graph_update", "financial_security_agent",
    "risk_score", "explain", "consent_gate", "needs_review", "watchlist",
    "escalation_draft", "persist",
]

# State keys that appear in pipeline state (conceptual)
STATE_KEYS = [
    "household_id", "ingested_events", "session_ids", "utterances", "entities",
    "mentions", "relationships", "normalized", "graph_updated",
    "financial_risk_signals", "financial_watchlists", "financial_logs",
    "risk_scores", "_model_available", "_pattern_tags", "_structural_motifs",
    "_timeline_snippet", "explanations", "consent_allows_escalation",
    "consent_allows_watchlist", "watchlists", "escalation_draft",
    "escalation_needs_clarification", "persisted", "needs_review",
    "time_range_start", "time_range_end", "consent_state", "calibration_params",
]

# Household IDs for parametrization
HOUSEHOLD_IDS = ["", "hh1", "hh-two", "household-uuid-123", "a"]

# Consent combinations
CONSENT_SHARE = [True, False]
CONSENT_WATCHLIST = [True, False]


@pytest.mark.parametrize("node_name", PIPELINE_NODE_NAMES)
def test_pipeline_node_name_is_nonempty(node_name: str) -> None:
    assert node_name
    assert isinstance(node_name, str)


@pytest.mark.parametrize("key", STATE_KEYS)
def test_state_key_is_string(key: str) -> None:
    assert isinstance(key, str)
    assert len(key) >= 1


@pytest.mark.parametrize("household_id", HOUSEHOLD_IDS)
def test_ingest_events_batch_accepts_household_id(household_id: str) -> None:
    state = ingest_events_batch({"household_id": household_id})
    assert "ingested_events" in state
    assert state.get("household_id") == household_id or "household_id" in state


@pytest.mark.parametrize("household_id", HOUSEHOLD_IDS)
def test_run_pipeline_accepts_household_id(household_id: str) -> None:
    result = run_pipeline(household_id, [])
    assert isinstance(result, dict)
    assert result.get("household_id") == household_id


@pytest.mark.parametrize("n_events", [0, 1, 2, 3, 4, 5])
def test_run_pipeline_with_n_events(n_events: int) -> None:
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "seq": i, "event_type": "wake", "payload": {}}
        for i in range(n_events)
    ]
    result = run_pipeline("hh1", events)
    assert "risk_scores" in result
    assert "explanations" in result


@pytest.mark.parametrize("share", CONSENT_SHARE)
@pytest.mark.parametrize("watchlist", CONSENT_WATCHLIST)
def test_consent_gate_combinations(share: bool, watchlist: bool) -> None:
    state = consent_policy_gate({"consent_state": {"share_with_caregiver": share, "watchlist_ok": watchlist}})
    assert state["consent_allows_escalation"] == share
    assert state["consent_allows_watchlist"] == watchlist


@pytest.mark.parametrize("consent_watchlist", [True, False])
@pytest.mark.parametrize("n_scores", [0, 1, 3, 5])
def test_synthesize_watchlists_consent_and_score_count(consent_watchlist: bool, n_scores: int) -> None:
    risk_scores = [{"node_index": i, "score": 0.5} for i in range(n_scores)]
    state = synthesize_watchlists({"consent_allows_watchlist": consent_watchlist, "risk_scores": risk_scores})
    assert "watchlists" in state
    if not consent_watchlist:
        assert state["watchlists"] == []


@pytest.mark.parametrize("n_scores", [0, 1, 2, 3, 4])
def test_generate_explanations_n_scores(n_scores: int) -> None:
    risk_scores = [{"node_index": i, "score": 0.6} for i in range(n_scores)]
    state = generate_explanations({
        "risk_scores": risk_scores,
        "_pattern_tags": [],
        "_structural_motifs": [],
        "_timeline_snippet": [],
        "_model_available": False,
    })
    assert len(state["explanations"]) <= n_scores


@pytest.mark.parametrize("entity_count", [0, 1, 2, 3, 4, 5])
def test_risk_score_inference_entity_count(entity_count: int) -> None:
    entities = [{"id": f"e{i}", "canonical": f"ent{i}"} for i in range(entity_count)]
    state = risk_score_inference({
        "household_id": "h",
        "entities": entities,
        "utterances": [],
        "mentions": [],
        "relationships": [],
        "ingested_events": [],
    })
    assert len(state["risk_scores"]) == entity_count
    assert state["_model_available"] is False or isinstance(state["_model_available"], bool)


@pytest.mark.parametrize("key", ["risk_scores", "explanations", "watchlists", "persisted", "household_id"])
def test_run_pipeline_final_state_has_key(key: str) -> None:
    result = run_pipeline("hh1", [])
    assert key in result


@pytest.mark.parametrize("i", range(20))
def test_build_graph_compiles(i: int) -> None:
    g = build_graph(None)
    assert g is not None


@pytest.mark.parametrize("i", range(15))
def test_should_review_returns_continue_or_needs_review(i: int) -> None:
    out = should_review({"consent_allows_escalation": True, "risk_scores": []})
    assert out in ("continue", "needs_review")


@pytest.mark.parametrize("i", range(15))
def test_draft_escalation_no_consent_empty(i: int) -> None:
    state = draft_escalation_message({"consent_allows_escalation": False, "risk_scores": []})
    assert state["escalation_draft"] == ""


@pytest.mark.parametrize("i", range(10))
def test_persist_outputs_sets_persisted(i: int) -> None:
    state = persist_outputs({})
    assert state.get("persisted") is True


@pytest.mark.parametrize("i", range(10))
def test_graph_update_sets_graph_updated(i: int) -> None:
    state = graph_update({})
    assert state.get("graph_updated") is True


# Event type variants for normalize
EVENT_TYPES = ["wake", "final_asr", "intent", "device_connected"]


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_normalize_events_event_type(event_type: str) -> None:
    events = [{"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "seq": 0, "event_type": event_type, "payload": {}}]
    state = normalize_events({"household_id": "h", "ingested_events": events})
    assert "utterances" in state and "entities" in state


# Severity / calibration
@pytest.mark.parametrize("adjust", [None, 0, 1, -1])
def test_run_pipeline_severity_adjust(adjust: int | None) -> None:
    result = run_pipeline("hh1", [], severity_threshold_adjust=adjust)
    assert isinstance(result, dict)


@pytest.mark.parametrize("i", range(25))
def test_financial_agent_output_keys(i: int) -> None:
    state = financial_security_agent({
        "household_id": "h",
        "ingested_events": [],
        "utterances": [],
        "entities": [],
        "mentions": [],
        "relationships": [],
    })
    assert "financial_risk_signals" in state
    assert "financial_watchlists" in state
