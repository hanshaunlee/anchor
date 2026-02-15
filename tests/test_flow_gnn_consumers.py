"""
Flow tests: where GNN outputs go. risk_scores (embedding, model_subgraph) feed:
- synthesize_watchlists -> embedding_centroid watchlist when >=3 high-risk with embeddings
- generate_explanations -> explanation_json.model_subgraph
- Worker persist -> risk_signal_embeddings table (conceptual); get_similar_incidents uses embeddings.
No internal code—only public API names and data flow assertions.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import synthesize_watchlists, generate_explanations, risk_score_inference


# --- GNN embeddings -> synthesize_watchlists: embedding_centroid watchlist ---
@pytest.mark.parametrize("n_embeddings", [3, 4, 5])
def test_embedding_centroid_watchlist_created_when_sufficient_high_risk_embeddings(n_embeddings: int) -> None:
    risk_scores = [
        {"node_index": i, "score": 0.7, "embedding": [0.1, 0.0, 0.0, 1.0]}
        for i in range(n_embeddings)
    ]
    state = synthesize_watchlists({
        "consent_allows_watchlist": True,
        "risk_scores": risk_scores,
    })
    centroid_wls = [w for w in state["watchlists"] if w.get("watch_type") == "embedding_centroid"]
    assert len(centroid_wls) == 1
    assert "pattern" in centroid_wls[0]
    assert centroid_wls[0]["pattern"].get("metric") == "cosine"
    assert "centroid" in centroid_wls[0]["pattern"]
    assert "provenance" in centroid_wls[0]["pattern"]


def test_no_embedding_centroid_when_fewer_than_three_embeddings() -> None:
    risk_scores = [
        {"node_index": 0, "score": 0.8, "embedding": [1.0, 0.0]},
        {"node_index": 1, "score": 0.8, "embedding": [0.0, 1.0]},
    ]
    state = synthesize_watchlists({"consent_allows_watchlist": True, "risk_scores": risk_scores})
    centroid_wls = [w for w in state["watchlists"] if w.get("watch_type") == "embedding_centroid"]
    assert len(centroid_wls) == 0


def test_no_embedding_centroid_when_scores_below_threshold() -> None:
    risk_scores = [
        {"node_index": i, "score": 0.2, "embedding": [0.1, 0.0, 0.0, 1.0]}
        for i in range(3)
    ]
    state = synthesize_watchlists({"consent_allows_watchlist": True, "risk_scores": risk_scores})
    centroid_wls = [w for w in state["watchlists"] if w.get("watch_type") == "embedding_centroid"]
    assert len(centroid_wls) == 0


# --- GNN model_subgraph -> generate_explanations ---
@pytest.mark.parametrize("has_model_subgraph", [True, False])
def test_explanation_includes_model_subgraph_when_available(has_model_subgraph: bool) -> None:
    subgraph = {"nodes": [{"id": "n1", "type": "entity"}], "edges": []} if has_model_subgraph else None
    risk_scores = [
        {"node_index": 0, "score": 0.6, "model_available": has_model_subgraph, "model_subgraph": subgraph},
    ]
    state = generate_explanations({
        "risk_scores": risk_scores,
        "_pattern_tags": [],
        "_structural_motifs": [],
        "_timeline_snippet": [],
        "_model_available": has_model_subgraph,
    })
    assert len(state["explanations"]) == 1
    ej = state["explanations"][0]["explanation_json"]
    assert "model_available" in ej
    if has_model_subgraph and subgraph:
        assert ej.get("model_subgraph") == subgraph


# --- risk_score_inference: when model runs, risk_scores get embedding key ---
def test_risk_score_inference_with_entities_returns_list_of_items() -> None:
    state = risk_score_inference({
        "household_id": "h",
        "entities": [{"id": "e1", "canonical": "alice"}],
        "utterances": [],
        "mentions": [],
        "relationships": [],
        "ingested_events": [],
    })
    assert isinstance(state["risk_scores"], list)
    if state["risk_scores"]:
        item = state["risk_scores"][0]
        assert "node_index" in item
        assert "score" in item
        if state.get("_model_available") and "embedding" in item:
            assert isinstance(item["embedding"], (list, tuple))


# --- Flow: pipeline run -> risk_scores -> watchlists and explanations ---
def test_full_pipeline_risk_scores_flow_into_watchlists_and_explanations() -> None:
    from api.pipeline import run_pipeline
    events = [
        {"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T00:00:00Z", "seq": 0, "event_type": "wake", "payload": {}},
    ]
    result = run_pipeline("hh-flow", events)
    assert "risk_scores" in result
    assert "watchlists" in result
    assert "explanations" in result
    assert isinstance(result["risk_scores"], list)
    assert isinstance(result["watchlists"], list)
    assert isinstance(result["explanations"], list)
