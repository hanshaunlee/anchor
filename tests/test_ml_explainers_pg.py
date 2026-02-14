"""Tests for ml.explainers.pg_explainer: PGExplainerStyle, explain_with_pg."""
import pytest

pytest.importorskip("torch")
import torch
from torch_geometric.data import Data

from ml.explainers.pg_explainer import PGExplainerStyle, explain_with_pg


def test_pg_explainer_style_forward() -> None:
    ex = PGExplainerStyle(hidden_dim=8, edge_attr_dim=0)
    node_emb = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    w = ex.forward(node_emb, edge_index, None)
    assert w.shape == (3,)


def test_pg_explainer_style_forward_with_edge_attr() -> None:
    ex = PGExplainerStyle(hidden_dim=4, edge_attr_dim=2)
    node_emb = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.randn(2, 2)
    w = ex.forward(node_emb, edge_index, edge_attr)
    assert w.shape == (2,)


def test_pg_explainer_style_edge_weights() -> None:
    ex = PGExplainerStyle(hidden_dim=8)
    node_emb = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    w = ex.edge_weights(node_emb, edge_index, None)
    assert w.shape == (2,)
    assert (w >= 0).all() and (w <= 1).all()


def test_explain_with_pg() -> None:
    ex = PGExplainerStyle(hidden_dim=8)
    node_emb = torch.randn(5, 8)
    data = Data(x=torch.randn(5, 8), edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    result = explain_with_pg(ex, node_emb, data, top_k=5)
    assert isinstance(result, dict)
    assert result["method"] == "pg_explainer"
    assert "top_edges" in result
    assert "minimal_subgraph_node_ids" in result
    assert "minimal_subgraph_edges" in result
    assert len(result["top_edges"]) <= 5


def test_explain_with_pg_with_edge_attr() -> None:
    ex = PGExplainerStyle(hidden_dim=4, edge_attr_dim=2)
    node_emb = torch.randn(3, 4)
    data = Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 2),
    )
    result = explain_with_pg(ex, node_emb, data, top_k=10)
    assert "top_edges" in result
    assert result["summary"].startswith("PGExplainer")
