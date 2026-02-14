"""Unit tests for pipeline nodes (ingest, normalize, risk_score, consent_gate, watchlist, escalation, should_review)."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import (
    ingest_events_batch,
    normalize_events,
    risk_score_inference,
    generate_explanations,
    consent_policy_gate,
    synthesize_watchlists,
    draft_escalation_message,
    should_review,
    graph_update,
    persist_outputs,
)


def test_ingest_events_batch_prefilled() -> None:
    state = {
        "household_id": "hh1",
        "time_range_start": "2024-01-01",
        "time_range_end": "2024-01-02",
        "ingested_events": [{"session_id": "s1", "event_type": "wake"}],
        "session_ids": ["s1"],
    }
    out = ingest_events_batch(state)
    assert out["ingested_events"]
    assert "s1" in out["session_ids"]
    assert "logs" in out and len(out["logs"]) >= 1


def test_ingest_events_batch_empty() -> None:
    state = {"household_id": "hh1", "time_range_start": None, "time_range_end": None}
    out = ingest_events_batch(state)
    assert out["ingested_events"] == []
    assert out["session_ids"] == []


def test_normalize_events() -> None:
    state = {
        "household_id": "hh1",
        "ingested_events": [
            {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
            {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:00Z", "seq": 1, "event_type": "final_asr", "payload": {"text": "Call Mom", "confidence": 0.9}},
            {"session_id": "s1", "device_id": "d1", "ts": "2024-01-01T10:01:01Z", "seq": 2, "event_type": "intent", "payload": {"name": "call", "slots": {"contact": "Mom"}, "confidence": 0.85}},
        ],
    }
    out = normalize_events(state)
    assert out["normalized"] is True
    assert "utterances" in out and len(out["utterances"]) >= 1
    assert "entities" in out
    assert "mentions" in out
    assert "relationships" in out
    assert "logs" in out


def test_graph_update() -> None:
    state = {"household_id": "hh1"}
    out = graph_update(state)
    assert out["graph_updated"] is True


def test_risk_score_inference_empty_entities() -> None:
    state = {"entities": [], "ingested_events": []}
    out = risk_score_inference(state)
    assert out["risk_scores"] == []


def test_risk_score_inference_with_entities() -> None:
    state = {
        "entities": [{"id": "e1"}, {"id": "e2"}],
        "ingested_events": [{"ts": "2024-01-01T10:00:00Z"}],
    }
    out = risk_score_inference(state)
    assert len(out["risk_scores"]) == 2
    for r in out["risk_scores"]:
        assert "node_index" in r
        assert "score" in r
        assert r["node_type"] == "entity"


def test_consent_policy_gate_defaults() -> None:
    state = {"consent_state": {}}
    out = consent_policy_gate(state)
    assert "consent_allows_escalation" in out
    assert "consent_allows_watchlist" in out
    assert out["consent_allows_escalation"] is True
    assert out["consent_allows_watchlist"] is True


def test_consent_policy_gate_explicit_false() -> None:
    state = {"consent_state": {"share_with_caregiver": False, "watchlist_ok": False}}
    out = consent_policy_gate(state)
    assert out["consent_allows_escalation"] is False
    assert out["consent_allows_watchlist"] is False


def test_synthesize_watchlists_no_consent() -> None:
    state = {"consent_allows_watchlist": False, "risk_scores": [{"node_index": 0, "score": 0.9}]}
    out = synthesize_watchlists(state)
    assert out["watchlists"] == []


def test_synthesize_watchlists_with_consent() -> None:
    state = {
        "consent_allows_watchlist": True,
        "risk_scores": [
            {"node_index": 0, "score": 0.1},
            {"node_index": 1, "score": 0.6},
        ],
    }
    out = synthesize_watchlists(state)
    assert len(out["watchlists"]) >= 1
    assert any(w["pattern"].get("node_index") == 1 for w in out["watchlists"])


def test_draft_escalation_message_no_consent() -> None:
    state = {"consent_allows_escalation": False, "risk_scores": [{"score": 0.9}]}
    out = draft_escalation_message(state)
    assert out["escalation_draft"] == ""


def test_draft_escalation_message_high_risk() -> None:
    state = {
        "consent_allows_escalation": True,
        "risk_scores": [{"score": 0.7}, {"score": 0.8}],
    }
    out = draft_escalation_message(state)
    assert "escalation" in out["escalation_draft"].lower()
    assert "high" in out["escalation_draft"].lower() or "2" in out["escalation_draft"]


def test_persist_outputs() -> None:
    state = {}
    out = persist_outputs(state)
    assert out["persisted"] is True


def test_should_review_no_consent() -> None:
    state = {"consent_allows_escalation": False, "risk_scores": [{"score": 0.99}]}
    assert should_review(state) == "continue"


def test_should_review_low_scores() -> None:
    state = {"consent_allows_escalation": True, "risk_scores": [{"score": 0.1}]}
    assert should_review(state) == "continue"


def test_should_review_high_severity() -> None:
    # severity = 1 + score*4; need >= 4 so score >= 0.75
    state = {"consent_allows_escalation": True, "risk_scores": [{"score": 0.8}]}
    assert should_review(state) == "needs_review"


def test_should_review_respects_calibration_adjust() -> None:
    # Base threshold 4; adjust +1 -> effective 5. Severity for score 0.8 is 4, so 4 >= 5 is False -> continue
    state = {"consent_allows_escalation": True, "risk_scores": [{"score": 0.8}], "severity_threshold_adjust": 1.0}
    assert should_review(state) == "continue"


def test_generate_explanations_model_available_false_when_no_model() -> None:
    state = {
        "_model_available": False,
        "risk_scores": [{"node_index": 0, "score": 0.6}],
        "utterances": [],
        "mentions": [],
        "entities": [{"id": "e0"}],
        "relationships": [],
        "ingested_events": [],
    }
    out = generate_explanations(state)
    assert len(out["explanations"]) >= 1
    expl = out["explanations"][0]["explanation_json"]
    assert expl["model_available"] is False
    # When model did not run: must not include model_subgraph.
    assert "model_subgraph" not in expl


def test_synthesize_watchlists_no_centroid_without_embeddings() -> None:
    state = {
        "consent_allows_watchlist": True,
        "risk_scores": [{"node_index": 0, "score": 0.7}, {"node_index": 1, "score": 0.6}],
    }
    out = synthesize_watchlists(state)
    centroid_wls = [w for w in out["watchlists"] if w.get("watch_type") == "embedding_centroid"]
    assert len(centroid_wls) == 0


def test_synthesize_watchlists_centroid_requires_min_three_embeddings() -> None:
    """Embedding-centroid watchlist only when >= 3 high-risk nodes have real embeddings."""
    state = {
        "consent_allows_watchlist": True,
        "risk_scores": [
            {"node_index": 0, "score": 0.7, "embedding": [0.1, 0.0, 0.0, 1.0]},
            {"node_index": 1, "score": 0.6, "embedding": [1.0, 0.0, 0.0, 0.0]},
        ],
    }
    out = synthesize_watchlists(state)
    centroid_wls = [w for w in out["watchlists"] if w.get("watch_type") == "embedding_centroid"]
    assert len(centroid_wls) == 0


def test_synthesize_watchlists_centroid_when_embeddings_present() -> None:
    state = {
        "consent_allows_watchlist": True,
        "risk_scores": [
            {"node_index": 0, "score": 0.7, "embedding": [0.1, 0.0, 0.0, 1.0]},
            {"node_index": 1, "score": 0.6, "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"node_index": 2, "score": 0.55, "embedding": [0.0, 1.0, 0.0, 0.0]},
        ],
    }
    out = synthesize_watchlists(state)
    centroid_wls = [w for w in out["watchlists"] if w.get("watch_type") == "embedding_centroid"]
    assert len(centroid_wls) == 1
    assert "centroid" in centroid_wls[0]["pattern"]
    assert centroid_wls[0]["pattern"].get("metric") == "cosine"
    assert centroid_wls[0]["pattern"].get("threshold") == 0.82
    assert "provenance" in centroid_wls[0]["pattern"]
    assert centroid_wls[0]["pattern"]["provenance"].get("window_days") == 14
    assert centroid_wls[0].get("expires_at_days") == 7
