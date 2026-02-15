"""
Flow tests: pipeline node state in/out. No internal code references—only public function names and state keys.
Data flow: ingest -> normalize -> graph_update -> financial_security_agent -> risk_score -> explain -> consent_gate -> watchlist -> escalation_draft -> persist.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import (
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
    needs_review_node,
    should_review,
    build_graph,
    run_pipeline,
)


# --- ingest_events_batch(state) -> state with ingested_events, session_ids ---
@pytest.mark.parametrize("household_id", ["", "hh1", "hh-two"])
def test_ingest_events_batch_accepts_household_id(household_id: str) -> None:
    state = ingest_events_batch({"household_id": household_id})
    assert "ingested_events" in state
    assert "session_ids" in state


@pytest.mark.parametrize("prefill_events", [[], [{"session_id": "s1", "ts": "2025-01-01T00:00:00Z"}]])
def test_ingest_events_batch_preserves_prefilled_events(prefill_events: list) -> None:
    state = ingest_events_batch({"household_id": "h", "ingested_events": prefill_events})
    assert state.get("ingested_events") == prefill_events


def test_ingest_events_batch_with_time_range() -> None:
    state = ingest_events_batch({
        "household_id": "h",
        "time_range_start": "2025-01-01",
        "time_range_end": "2025-01-02",
    })
    assert "ingested_events" in state


# --- normalize_events(state) -> utterances, entities, mentions, relationships, normalized ---
@pytest.mark.parametrize("event_count", [0, 1, 3])
def test_normalize_events_output_keys(event_count: int) -> None:
    events = [{"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "seq": i, "event_type": "wake", "payload": {}} for i in range(event_count)]
    state = normalize_events({"household_id": "h", "ingested_events": events})
    assert "utterances" in state
    assert "entities" in state
    assert "mentions" in state
    assert "relationships" in state
    assert state.get("normalized") is True


def test_normalize_events_empty_events() -> None:
    state = normalize_events({"household_id": "h", "ingested_events": []})
    assert isinstance(state["utterances"], list)
    assert isinstance(state["entities"], list)


# --- graph_update(state) -> graph_updated ---
def test_graph_update_sets_graph_updated() -> None:
    state = graph_update({"household_id": "h"})
    assert state.get("graph_updated") is True


# --- financial_security_agent(state) -> financial_risk_signals, financial_watchlists, financial_logs ---
@pytest.mark.parametrize("has_events", [True, False])
def test_financial_security_agent_output_keys(has_events: bool) -> None:
    events = [{"session_id": "s1", "ts": "2025-01-01T00:00:00Z", "event_type": "wake", "payload": {}}] if has_events else []
    state = financial_security_agent({
        "household_id": "h",
        "ingested_events": events,
        "utterances": [],
        "entities": [],
        "mentions": [],
        "relationships": [],
    })
    assert "financial_risk_signals" in state
    assert "financial_watchlists" in state
    assert "financial_logs" in state


# --- risk_score_inference(state) -> risk_scores, _model_available ---
@pytest.mark.parametrize("entity_count", [0, 1, 5])
def test_risk_score_inference_output_keys(entity_count: int) -> None:
    entities = [{"id": f"e{i}", "canonical": f"entity{i}"} for i in range(entity_count)]
    state = risk_score_inference({
        "household_id": "h",
        "ingested_events": [],
        "entities": entities,
        "utterances": [],
        "mentions": [],
        "relationships": [],
    })
    assert "risk_scores" in state
    assert "_model_available" in state
    assert isinstance(state["risk_scores"], list)


def test_risk_score_inference_empty_entities_sets_model_available_false() -> None:
    state = risk_score_inference({"household_id": "h", "entities": [], "ingested_events": []})
    assert state["_model_available"] is False
    assert state["risk_scores"] == []


# --- generate_explanations(state) -> explanations ---
@pytest.mark.parametrize("score_count", [0, 1, 4])
def test_generate_explanations_output_keys(score_count: int) -> None:
    risk_scores = [{"node_index": i, "score": 0.5, "model_available": False} for i in range(score_count)]
    state = generate_explanations({
        "risk_scores": risk_scores,
        "_pattern_tags": [],
        "_structural_motifs": [],
        "_timeline_snippet": [],
        "_model_available": False,
    })
    assert "explanations" in state
    assert isinstance(state["explanations"], list)


def test_generate_explanations_each_has_explanation_json() -> None:
    state = generate_explanations({
        "risk_scores": [{"node_index": 0, "score": 0.6}],
        "_pattern_tags": [],
        "_structural_motifs": [],
        "_timeline_snippet": [],
        "_model_available": False,
    })
    for e in state["explanations"]:
        assert "explanation_json" in e
        assert "node_index" in e


# --- consent_policy_gate(state) -> consent_allows_escalation, consent_allows_watchlist ---
@pytest.mark.parametrize("consent_share", [True, False])
@pytest.mark.parametrize("consent_watchlist", [True, False])
def test_consent_policy_gate_sets_flags(consent_share: bool, consent_watchlist: bool) -> None:
    state = consent_policy_gate({
        "consent_state": {"share_with_caregiver": consent_share, "watchlist_ok": consent_watchlist},
    })
    assert "consent_allows_escalation" in state
    assert "consent_allows_watchlist" in state
    assert state["consent_allows_escalation"] == consent_share
    assert state["consent_allows_watchlist"] == consent_watchlist


# --- synthesize_watchlists(state) -> watchlists ---
@pytest.mark.parametrize("consent_allows_watchlist", [True, False])
def test_synthesize_watchlists_output(consent_allows_watchlist: bool) -> None:
    state = synthesize_watchlists({
        "consent_allows_watchlist": consent_allows_watchlist,
        "risk_scores": [{"node_index": 0, "score": 0.7}],
    })
    assert "watchlists" in state
    assert isinstance(state["watchlists"], list)
    if not consent_allows_watchlist:
        assert state["watchlists"] == []


def test_synthesize_watchlists_high_score_produces_entity_pattern() -> None:
    state = synthesize_watchlists({
        "consent_allows_watchlist": True,
        "risk_scores": [{"node_index": 0, "score": 0.8}],
    })
    assert len(state["watchlists"]) >= 1
    assert any(w.get("watch_type") == "entity_pattern" for w in state["watchlists"])


def test_synthesize_watchlists_low_score_no_entity_pattern() -> None:
    state = synthesize_watchlists({
        "consent_allows_watchlist": True,
        "risk_scores": [{"node_index": 0, "score": 0.1}],
    })
    entity_wls = [w for w in state["watchlists"] if w.get("watch_type") == "entity_pattern"]
    assert len(entity_wls) == 0


# --- draft_escalation_message(state) -> escalation_draft, escalation_needs_clarification ---
def test_draft_escalation_message_output_keys() -> None:
    state = draft_escalation_message({
        "consent_allows_escalation": True,
        "risk_scores": [],
    })
    assert "escalation_draft" in state
    assert "escalation_needs_clarification" in state


def test_draft_escalation_message_no_consent_empty_draft() -> None:
    state = draft_escalation_message({"consent_allows_escalation": False, "risk_scores": []})
    assert state["escalation_draft"] == ""


# --- persist_outputs(state) -> persisted ---
def test_persist_outputs_sets_persisted() -> None:
    state = persist_outputs({})
    assert state.get("persisted") is True


# --- needs_review_node(state) -> needs_review ---
def test_needs_review_node_sets_needs_review() -> None:
    state = needs_review_node({})
    assert state.get("needs_review") is True


# --- should_review(state) -> "needs_review" | "continue" ---
@pytest.mark.parametrize("consent", [True, False])
def test_should_review_returns_continue_when_no_consent(consent: bool) -> None:
    out = should_review({"consent_allows_escalation": consent, "risk_scores": []})
    assert out in ("needs_review", "continue")
    if not consent:
        assert out == "continue"


# --- build_graph(checkpointer) -> StateGraph ---
@pytest.mark.parametrize("with_checkpointer", [True, False])
def test_build_graph_returns_compiled(with_checkpointer: bool) -> None:
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver() if with_checkpointer else None
    g = build_graph(checkpointer)
    assert g is not None
    assert hasattr(g, "invoke") or hasattr(g, "stream")


# --- run_pipeline(household_id, ingested_events, ...) -> final state ---
def test_run_pipeline_returns_dict() -> None:
    result = run_pipeline("hh1", [])
    assert isinstance(result, dict)


def test_run_pipeline_final_state_has_core_keys() -> None:
    result = run_pipeline("hh1", [])
    assert "household_id" in result
    assert "risk_scores" in result
    assert "explanations" in result
    assert "watchlists" in result
    assert "persisted" in result


@pytest.mark.parametrize("event_count", [0, 1, 2])
def test_run_pipeline_with_events_event_count(event_count: int) -> None:
    events = [{"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "seq": i, "event_type": "wake", "payload": {}} for i in range(event_count)]
    result = run_pipeline("hh1", events)
    assert "ingested_events" in result or "session_ids" in result
    assert "risk_scores" in result


def test_run_pipeline_accepts_calibration_params() -> None:
    result = run_pipeline("hh1", [], calibration_params={"platt_a": 1.0, "platt_b": 0.0})
    assert isinstance(result, dict)


def test_run_pipeline_accepts_severity_threshold_adjust() -> None:
    result = run_pipeline("hh1", [], severity_threshold_adjust=1)
    assert isinstance(result, dict)
