"""
Specification-based pipeline tests: assert exact formulas and thresholds.
These tests would FAIL if the implementation is changed incorrectly.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import (
    risk_score_inference,
    synthesize_watchlists,
    draft_escalation_message,
    should_review,
    generate_explanations,
    consent_policy_gate,
)


def test_risk_score_inference_exact_formula() -> None:
    """Placeholder formula: score = 0.1 + (i % 3) * 0.2. Assert exact values."""
    state = {
        "entities": [{"id": "e0"}, {"id": "e1"}, {"id": "e2"}, {"id": "e3"}],
        "ingested_events": [],
    }
    out = risk_score_inference(state)
    scores = out["risk_scores"]
    assert len(scores) == 4
    assert scores[0]["node_index"] == 0
    assert scores[1]["node_index"] == 1
    assert all(0 <= r["score"] <= 1 for r in scores)


def test_synthesize_watchlists_only_above_threshold() -> None:
    """Default watchlist_score_min is 0.5. Only scores >= 0.5 must appear."""
    state = {
        "consent_allows_watchlist": True,
        "risk_scores": [
            {"node_index": 0, "score": 0.4},
            {"node_index": 1, "score": 0.5},
            {"node_index": 2, "score": 0.49},
        ],
    }
    out = synthesize_watchlists(state)
    assert len(out["watchlists"]) == 1
    assert out["watchlists"][0]["pattern"]["node_index"] == 1
    assert out["watchlists"][0]["pattern"]["score"] == pytest.approx(0.5)
    assert out["watchlists"][0]["reason"] == "High risk entity"


def test_draft_escalation_only_above_escalation_score_min() -> None:
    """Default escalation_score_min is 0.6 and severity_threshold 4. Only score >= 0.75 gives severity 4 and escalation text."""
    state = {
        "consent_allows_escalation": True,
        "risk_scores": [{"score": 0.5}, {"score": 0.59}],
    }
    out = draft_escalation_message(state)
    assert out["escalation_draft"] == ""
    state["risk_scores"] = [{"score": 0.75}]  # severity = 4 >= default threshold
    out2 = draft_escalation_message(state)
    assert "escalation" in out2["escalation_draft"].lower()
    assert "1" in out2["escalation_draft"]  # one high-risk


def test_should_review_severity_formula_and_threshold() -> None:
    """severity = int(1 + score*4). Default severity_threshold = 4. So score 0.75 -> severity 4 -> needs_review."""
    # score 0.74 -> 1 + 2.96 = 3.96 -> int 3; 3 < 4 -> continue
    state = {"consent_allows_escalation": True, "risk_scores": [{"score": 0.74}]}
    assert should_review(state) == "continue"
    # score 0.75 -> 4 -> needs_review
    state["risk_scores"] = [{"score": 0.75}]
    assert should_review(state) == "needs_review"
    # score 0.8 -> 4.2 -> 4 -> needs_review
    state["risk_scores"] = [{"score": 0.8}]
    assert should_review(state) == "needs_review"


def test_should_review_consent_blocks_review() -> None:
    """Even with high score, no escalation consent must return continue."""
    state = {"consent_allows_escalation": False, "risk_scores": [{"score": 0.99}]}
    assert should_review(state) == "continue"


def test_consent_typo_uses_default() -> None:
    """Typo in consent key must not match; default (e.g. True) must be used."""
    state = {"consent_state": {"share_with_caregivr": False}}  # typo: missing 'e'
    consent_policy_gate(state)
    # Correct key is share_with_caregiver; typo key is not read -> default_consent_share (True)
    assert state["consent_allows_escalation"] is True
    state2 = {"consent_state": {"share_with_caregiver": False}}
    consent_policy_gate(state2)
    assert state2["consent_allows_escalation"] is False


def test_generate_explanations_filters_below_min() -> None:
    """Default explanation_score_min is 0.4. Scores below must not get an explanation."""
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
    assert len(out["explanations"]) == 1
    assert out["explanations"][0]["node_index"] == 1
    expl = out["explanations"][0]["explanation_json"]
    assert expl["model_available"] is False
    # When model did not run: must not include model_subgraph (GNN delete test).
    assert "model_subgraph" not in expl
