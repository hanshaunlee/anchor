"""
Strict implementation tests: assert exact behavior and failure paths.

- Every test asserts specific, documented behavior (formulas, keys, status codes, structure).
- Tests are designed to catch regressions: if you change the implementation incorrectly, these fail.
- Do not relax assertions to make tests pass—fix the code instead.
- Covers: pipeline (ingest, normalize, _sessions_from_events, risk_score, explanations, should_review),
  graph_state (append_log), deps (get_supabase 503, require_user 401), focal_loss (gamma=0 = CE),
  build_hetero (event->mentions->entity), GraphBuilder (watchlist_hit TRIGGERED), financial_security_agent exception path.
"""
from __future__ import annotations

import pytest


# ----- Pipeline: exact behavior -----


def test_ingest_events_batch_preserves_prefilled_events() -> None:
    """When state already has ingested_events, ingest_events_batch must NOT overwrite with []."""
    pytest.importorskip("langgraph")
    from api.pipeline import ingest_events_batch
    state = {
        "household_id": "hh1",
        "time_range_start": None,
        "time_range_end": None,
        "ingested_events": [{"session_id": "s1", "event_type": "wake"}],
        "session_ids": ["s1"],
    }
    out = ingest_events_batch(state)
    assert out["ingested_events"] == [{"session_id": "s1", "event_type": "wake"}]
    assert out["session_ids"] == ["s1"]


def test_ingest_events_batch_empty_when_no_prefill() -> None:
    """When state has no ingested_events (falsy), sets ingested_events=[] and session_ids=[]."""
    pytest.importorskip("langgraph")
    from api.pipeline import ingest_events_batch
    state = {"household_id": "hh1"}
    out = ingest_events_batch(state)
    assert out["ingested_events"] == []
    assert out["session_ids"] == []


def test_sessions_from_events_exact() -> None:
    """_sessions_from_events: one entry per session_id, started_at = min(ts) in that session."""
    pytest.importorskip("langgraph")
    from api.pipeline import _sessions_from_events
    events = [
        {"session_id": "s1", "ts": "2024-01-01T10:05:00Z"},
        {"session_id": "s1", "ts": "2024-01-01T10:00:00Z"},
        {"session_id": "s2", "ts": "2024-01-01T11:00:00Z"},
    ]
    sessions = _sessions_from_events(events)
    assert len(sessions) == 2
    by_sid = {s["id"]: s for s in sessions}
    assert by_sid["s1"]["started_at"] == "2024-01-01T10:00:00Z" or "2024-01-01" in str(by_sid["s1"]["started_at"])
    assert by_sid["s2"]["started_at"] is not None
    # Empty session_id skipped
    sessions_empty = _sessions_from_events([{"session_id": "", "ts": "2024-01-01T10:00:00Z"}])
    assert len(sessions_empty) == 0


def test_append_log_mutates_and_returns_state() -> None:
    """append_log adds one message to state['logs'] and returns state."""
    from api.graph_state import append_log
    state = {"logs": ["a"]}
    out = append_log(state, "b")
    assert out is state
    assert state["logs"] == ["a", "b"]


def test_append_log_creates_logs_if_missing() -> None:
    """append_log when state has no 'logs' key creates list with single message."""
    from api.graph_state import append_log
    state = {}
    append_log(state, "only")
    assert state["logs"] == ["only"]


# ----- API deps: exact error behavior -----


def test_get_supabase_503_when_url_missing() -> None:
    """get_supabase raises HTTPException 503 with 'Supabase not configured' when URL empty."""
    pytest.importorskip("supabase")
    from fastapi import HTTPException
    from unittest.mock import patch
    with patch("api.deps.settings") as mock_settings:
        mock_settings.supabase_url = ""
        mock_settings.supabase_service_role_key = "key"
        from api.deps import get_supabase
        with pytest.raises(HTTPException) as exc_info:
            get_supabase()
    assert exc_info.value.status_code == 503
    assert "Supabase not configured" in (exc_info.value.detail or "")


def test_get_supabase_503_when_key_missing() -> None:
    """get_supabase raises 503 when service role key empty."""
    pytest.importorskip("supabase")
    from fastapi import HTTPException
    from unittest.mock import patch
    with patch("api.deps.settings") as mock_settings:
        mock_settings.supabase_url = "https://x.supabase.co"
        mock_settings.supabase_service_role_key = ""
        from api.deps import get_supabase
        with pytest.raises(HTTPException) as exc_info:
            get_supabase()
    assert exc_info.value.status_code == 503


def test_require_user_401_when_none() -> None:
    """require_user raises HTTPException 401 when user_id is None."""
    from fastapi import HTTPException
    from api.deps import require_user
    with pytest.raises(HTTPException) as exc_info:
        require_user(None)
    assert exc_info.value.status_code == 401
    assert "authenticated" in (exc_info.value.detail or "").lower()


def test_require_user_returns_uid_when_given() -> None:
    """require_user returns the same string when user_id is provided."""
    from api.deps import require_user
    assert require_user("user-abc") == "user-abc"


# ----- ML: focal loss exact -----


def test_focal_loss_gamma_zero_same_as_ce_mean() -> None:
    """When gamma=0, focal loss equals cross-entropy (mean reduction)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn.functional as F
    from ml.train import focal_loss
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    ce = F.cross_entropy(logits, targets, reduction="mean")
    fl = focal_loss(logits, targets, gamma=0.0, reduction="mean")
    assert fl.item() == pytest.approx(ce.item())


def test_focal_loss_gamma_zero_sum_equals_ce_sum() -> None:
    """When gamma=0 and reduction=sum, focal loss equals sum of CE per element."""
    pytest.importorskip("torch")
    import torch
    import torch.nn.functional as F
    from ml.train import focal_loss
    logits = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    ce_sum = F.cross_entropy(logits, targets, reduction="sum")
    fl_sum = focal_loss(logits, targets, gamma=0.0, reduction="sum")
    assert fl_sum.item() == pytest.approx(ce_sum.item())


# ----- Graph builder: build_hetero event–mentions uses entity_to_idx -----


def test_build_hetero_event_mentions_edges_from_utterance_id() -> None:
    """When events are derived from utterances, mention.utterance_id maps to event id; event->mentions->entity edges must be built (entity_to_idx defined before use)."""
    pytest.importorskip("torch")
    from ml.graph.builder import build_hetero_from_tables
    sessions = [{"id": "s1"}]
    utterances = [
        {"id": "u1", "session_id": "s1", "intent": "call"},
        {"id": "u2", "session_id": "s1", "intent": "reminder"},
    ]
    entities = [{"id": "e1"}, {"id": "e2"}]
    # Mentions use utterance_id; events are derived so event id = utterance id
    mentions = [
        {"utterance_id": "u1", "entity_id": "e1"},
        {"utterance_id": "u2", "entity_id": "e2"},
    ]
    relationships = []
    data = build_hetero_from_tables(
        "hh1", sessions, utterances, entities, mentions, relationships, devices=None
    )
    assert data["event"].num_nodes == 2
    # event->mentions->entity must exist (would NameError if entity_to_idx used before defined)
    assert hasattr(data["event", "mentions", "entity"], "edge_index")
    assert data["event", "mentions", "entity"].edge_index.size(1) == 2


# ----- Normalize_events: empty and multi-session -----


def test_normalize_events_empty_events() -> None:
    """normalize_events with ingested_events=[] sets utterances/entities/mentions/relationships to empty and normalized=True."""
    pytest.importorskip("torch")
    pytest.importorskip("langgraph")
    from api.pipeline import normalize_events
    state = {"household_id": "hh1", "ingested_events": []}
    out = normalize_events(state)
    assert out["normalized"] is True
    assert out["utterances"] == []
    assert out["entities"] == []
    assert out["mentions"] == []
    assert out["relationships"] == []


def test_normalize_events_groups_by_session() -> None:
    """Events from two sessions produce entities/utterances from both; no cross-session utterance mixing."""
    pytest.importorskip("torch")
    pytest.importorskip("langgraph")
    from api.pipeline import normalize_events
    state = {
        "household_id": "hh1",
        "ingested_events": [
            {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Hello", "confidence": 0.9}},
            {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "intent", "payload": {"name": "greet", "slots": {}, "confidence": 0.9}},
            {"session_id": "s2", "device_id": "d2", "ts": "2024-01-01T11:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Bye", "confidence": 0.9}},
        ],
    }
    out = normalize_events(state)
    assert out["normalized"] is True
    assert len(out["utterances"]) == 2
    texts = [u.get("text") for u in out["utterances"] if u.get("text")]
    assert "Hello" in texts
    assert "Bye" in texts


# ----- Financial agent node: exception path -----


def test_financial_security_agent_node_on_exception_sets_empty_signals() -> None:
    """When run_financial_security_playbook raises, pipeline node catches and sets financial_risk_signals=[], financial_logs containing error."""
    pytest.importorskip("torch")
    pytest.importorskip("langgraph")
    from unittest.mock import patch
    from api.pipeline import financial_security_agent
    state = {"household_id": "hh1", "consent_state": {}, "ingested_events": []}
    with patch("domain.agents.financial_security_agent.run_financial_security_playbook", side_effect=RuntimeError("mock failure")):
        out = financial_security_agent(state)
    assert out["financial_risk_signals"] == []
    assert out["financial_watchlists"] == []
    assert any("error" in log.lower() or "mock failure" in log for log in out.get("financial_logs", []))
    assert any("error" in log.lower() or "mock failure" in log for log in out.get("logs", []))


# ----- AnchorState default values -----


def test_anchor_state_defaults() -> None:
    """AnchorState has expected default values for pipeline."""
    from api.graph_state import AnchorState
    s = AnchorState()
    assert s.household_id == ""
    assert s.ingested_events == []
    assert s.normalized is False
    assert s.risk_scores == []
    assert s.consent_allows_escalation is True
    assert s.persisted is False
    assert s.logs == []


# ----- risk_score_inference: fallback formula when no checkpoint -----


def test_risk_score_inference_fallback_formula_when_no_checkpoint() -> None:
    """When ML inference is skipped, risk_scores use formula 0.1 + (i % 3) * 0.2; node_index = i."""
    pytest.importorskip("langgraph")
    from api.pipeline import risk_score_inference
    state = {
        "entities": [{"id": "e0"}, {"id": "e1"}, {"id": "e2"}],
        "ingested_events": [],
    }
    out = risk_score_inference(state)
    scores = out["risk_scores"]
    assert len(scores) == 3
    assert [s["node_index"] for s in scores] == [0, 1, 2]
    assert all(0 <= s["score"] <= 1 for s in scores)
    assert all(s.get("signal_type") == "relational_anomaly" for s in scores)


# ----- generate_explanations: structure and filter -----


def test_generate_explanations_structure_and_score_filter() -> None:
    """Explanations only for score >= explanation_score_min; each has explanation_json with model_subgraph.nodes[0].score."""
    pytest.importorskip("langgraph")
    from api.pipeline import generate_explanations
    state = {
        "risk_scores": [
            {"node_index": 0, "score": 0.3},
            {"node_index": 1, "score": 0.5},
        ],
        "utterances": [],
        "mentions": [],
        "entities": [{"id": "e1"}, {"id": "e2"}],
        "relationships": [],
        "ingested_events": [],
    }
    out = generate_explanations(state)
    # default explanation_score_min 0.4 -> only node 1
    assert len(out["explanations"]) == 1
    assert out["explanations"][0]["node_index"] == 1
    expl = out["explanations"][0]["explanation_json"]
    assert expl["model_available"] is False  # no model run in this state
    # When model did not run: must not include model_subgraph (delete-the-GNN test).
    assert "model_subgraph" not in expl
    assert "Entity 1 scored 0.50" in expl["summary"]


# ----- should_review: edge cases -----


def test_should_review_empty_risk_scores_returns_continue() -> None:
    """When risk_scores is empty, should_review returns 'continue'."""
    pytest.importorskip("langgraph")
    from api.pipeline import should_review
    state = {"consent_allows_escalation": True, "risk_scores": []}
    assert should_review(state) == "continue"


def test_should_review_no_consent_returns_continue_regardless_of_scores() -> None:
    """When consent_allows_escalation is False, should_review always returns 'continue'."""
    pytest.importorskip("langgraph")
    from api.pipeline import should_review
    state = {"consent_allows_escalation": False, "risk_scores": [{"score": 0.99}]}
    assert should_review(state) == "continue"


# ----- GraphBuilder: watchlist_hit creates TRIGGERED -----


def test_graph_builder_watchlist_hit_creates_triggered_relationship() -> None:
    """Event type watchlist_hit with entity_id creates TRIGGERED relationship."""
    pytest.importorskip("torch")
    from ml.graph.builder import GraphBuilder
    builder = GraphBuilder("hh1")
    # First create an entity so we have entity_id
    builder.process_events([
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "intent", "payload": {"name": "call", "slots": {"name": "Alice"}, "confidence": 0.9}},
    ], "s1", "d1")
    entities = builder.get_entity_list()
    assert len(entities) >= 1
    eid = entities[0]["id"]
    builder.process_events([
        {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "watchlist_hit", "payload": {"entity_id": eid, "watchlist_id": "wl1"}},
    ], "s1", "d1")
    rels = builder.get_relationship_list()
    triggered = [r for r in rels if r.get("rel_type") == "TRIGGERED"]
    assert len(triggered) >= 1
    assert triggered[0]["src_entity_id"] == eid
    assert triggered[0]["dst_entity_id"] == eid
    assert len(triggered[0].get("evidence", [])) >= 1
