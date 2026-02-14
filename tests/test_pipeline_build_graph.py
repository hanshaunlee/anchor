"""Tests for pipeline: build_graph, needs_review_node, generate_explanations."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from api.pipeline import (
    build_graph,
    needs_review_node,
    generate_explanations,
)


def test_build_graph_returns_compiled() -> None:
    graph = build_graph(checkpointer=None)
    assert graph is not None
    assert hasattr(graph, "invoke") or hasattr(graph, "stream")


def test_needs_review_node() -> None:
    state = {}
    out = needs_review_node(state)
    assert out["needs_review"] is True
    assert "logs" in out


def test_generate_explanations_empty_scores() -> None:
    state = {"risk_scores": [], "utterances": [], "mentions": [], "entities": [], "relationships": [], "ingested_events": []}
    out = generate_explanations(state)
    assert out["explanations"] == []


def test_generate_explanations_with_scores() -> None:
    state = {
        "risk_scores": [{"node_index": 0, "score": 0.5}, {"node_index": 1, "score": 0.3}],
        "utterances": [],
        "mentions": [],
        "entities": [{"id": "e1"}, {"id": "e2"}],
        "relationships": [],
        "ingested_events": [],
    }
    out = generate_explanations(state)
    assert len(out["explanations"]) >= 1
    for e in out["explanations"]:
        assert "node_index" in e
        assert "explanation_json" in e
        assert "motif_tags" in e["explanation_json"]
        assert "timeline_snippet" in e["explanation_json"]
