"""Comprehensive tests for domain.explain_service: build_subgraph_from_explanation and get_similar_incidents delegate."""
from __future__ import annotations

from uuid import uuid4
from unittest.mock import MagicMock

import pytest

from api.schemas import RiskSignalDetailSubgraph, SubgraphNode, SubgraphEdge
from domain.explain_service import build_subgraph_from_explanation, get_similar_incidents, run_deep_dive_explainer


def test_build_subgraph_from_explanation_model_subgraph() -> None:
    """Builds subgraph from explanation.model_subgraph when present."""
    expl = {
        "model_subgraph": {
            "nodes": [{"id": "e1", "type": "entity", "label": "Alice", "score": 0.8}],
            "edges": [{"src": "e1", "dst": "e2", "type": "mentions", "weight": 0.5, "rank": 0}],
        },
    }
    out = build_subgraph_from_explanation(expl)
    assert out is not None
    assert isinstance(out, RiskSignalDetailSubgraph)
    assert len(out.nodes) == 1
    assert out.nodes[0].id == "e1"
    assert out.nodes[0].type == "entity"
    assert out.nodes[0].label == "Alice"
    assert out.nodes[0].score == 0.8
    assert len(out.edges) == 1
    assert out.edges[0].src == "e1"
    assert out.edges[0].dst == "e2"
    assert out.edges[0].weight == 0.5
    assert out.edges[0].rank == 0


def test_build_subgraph_from_explanation_subgraph_fallback() -> None:
    """Uses explanation.subgraph when model_subgraph is absent."""
    expl = {
        "subgraph": {
            "nodes": [{"id": "n1", "type": "utterance"}],
            "edges": [],
        },
    }
    out = build_subgraph_from_explanation(expl)
    assert out is not None
    assert len(out.nodes) == 1
    assert out.nodes[0].id == "n1"
    assert out.nodes[0].type == "utterance"
    assert len(out.edges) == 0


def test_build_subgraph_from_explanation_empty_nodes_edges_returns_none() -> None:
    """When nodes and edges are both empty, returns None."""
    expl = {"model_subgraph": {"nodes": [], "edges": []}}
    out = build_subgraph_from_explanation(expl)
    assert out is None


def test_build_subgraph_from_explanation_missing_keys_returns_none() -> None:
    """When neither subgraph nor model_subgraph present or both empty, returns None."""
    out = build_subgraph_from_explanation({})
    assert out is None
    out = build_subgraph_from_explanation({"summary": "x"})
    assert out is None


def test_build_subgraph_from_explanation_node_defaults() -> None:
    """Node missing id/type uses empty string; missing label/score are None."""
    expl = {"model_subgraph": {"nodes": [{}], "edges": []}}
    out = build_subgraph_from_explanation(expl)
    assert out is not None
    assert out.nodes[0].id == ""
    assert out.nodes[0].type == ""
    assert out.nodes[0].label is None
    assert out.nodes[0].score is None


def test_build_subgraph_from_explanation_edge_defaults() -> None:
    """Edge without type gets empty string; weight and rank can be None."""
    expl = {
        "model_subgraph": {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"src": "a", "dst": "b"}],
        },
    }
    out = build_subgraph_from_explanation(expl)
    assert out is not None
    assert out.edges[0].type == ""
    assert out.edges[0].weight is None
    assert out.edges[0].rank is None


def test_build_subgraph_from_explanation_prefer_deep_dive() -> None:
    """When prefer_key=deep_dive_subgraph and present, uses it instead of model_subgraph."""
    expl = {
        "model_subgraph": {"nodes": [{"id": "a"}], "edges": []},
        "deep_dive_subgraph": {"nodes": [{"id": "b", "type": "entity"}], "edges": [{"src": "b", "dst": "c", "importance": 0.9}]},
    }
    out = build_subgraph_from_explanation(expl, prefer_key="deep_dive_subgraph")
    assert out is not None
    assert len(out.nodes) == 1 and out.nodes[0].id == "b"
    assert len(out.edges) == 1 and out.edges[0].weight == 0.9  # importance used as weight

def test_build_subgraph_from_explanation_edge_importance_fallback() -> None:
    """Edge with importance but no weight uses importance for weight."""
    expl = {"model_subgraph": {"nodes": [{"id": "x"}, {"id": "y"}], "edges": [{"src": "x", "dst": "y", "importance": 0.7}]}}
    out = build_subgraph_from_explanation(expl)
    assert out is not None and out.edges[0].weight == 0.7

def test_get_similar_incidents_delegates_to_similarity_service() -> None:
    """get_similar_incidents delegates to similarity_service; no row -> available=False."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.limit.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.return_value = q

    out = get_similar_incidents(uuid4(), "hh-1", mock_sb, top_k=3)
    assert out.available is False
    assert out.reason == "model_not_run"
    assert out.similar == []


def test_run_deep_dive_explainer_pg_success() -> None:
    """mode=pg copies model_subgraph to deep_dive_subgraph and persists."""
    signal_id = uuid4()
    household_id = "hh-dd"
    expl = {"model_subgraph": {"nodes": [{"id": "e1", "type": "entity"}], "edges": [{"src": "e1", "dst": "e2", "weight": 0.8}]}}
    mock_sb = MagicMock()
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.single.return_value = chain
    chain.execute.return_value.data = {"id": str(signal_id), "explanation": expl}
    mock_sb.table.return_value.select.return_value = chain
    mock_sb.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock()

    result = run_deep_dive_explainer(signal_id, household_id, mock_sb, mode="pg")
    assert result["ok"] is True
    assert result["method"] == "pg"
    assert "deep_dive_subgraph" in result
    assert result["deep_dive_subgraph"]["method"] == "pg"
    assert len(result["deep_dive_subgraph"]["nodes"]) == 1
    mock_sb.table.return_value.update.assert_called_once()
    call_expl = mock_sb.table.return_value.update.call_args[0][0]["explanation"]
    assert call_expl.get("deep_dive_subgraph", {}).get("method") == "pg"


def test_run_deep_dive_explainer_gnn_not_implemented() -> None:
    """mode=gnn raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="gnn"):
        run_deep_dive_explainer(uuid4(), "hh", MagicMock(), mode="gnn")


def test_run_deep_dive_explainer_no_model_subgraph_raises() -> None:
    """mode=pg when explanation has no model_subgraph raises ValueError."""
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute.return_value.data = {"id": "rs1", "explanation": {}}
    with pytest.raises(ValueError, match="No model subgraph"):
        run_deep_dive_explainer(uuid4(), "hh", mock_sb, mode="pg")
