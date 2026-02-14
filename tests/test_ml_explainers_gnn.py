"""Tests for ml.explainers.gnn_explainer: GNNExplainerStyle, explain_node_gnn_explainer_style."""
import pytest

pytest.importorskip("torch")
import torch
from torch_geometric.data import Data

from ml.explainers.gnn_explainer import GNNExplainerStyle, explain_node_gnn_explainer_style


def test_gnn_explainer_style_edge_mask() -> None:
    ex = GNNExplainerStyle(num_edges=5, num_features=0)
    mask = ex.get_edge_mask()
    assert mask.shape == (5,)
    assert (mask >= 0).all() and (mask <= 1).all()


def test_gnn_explainer_style_apply_masks() -> None:
    ex = GNNExplainerStyle(num_edges=3, num_features=4)
    data = Data(x=torch.randn(4, 4), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), edge_attr=torch.randn(3, 2))
    out = ex.apply_masks(data)
    assert out.x.shape == data.x.shape
    assert out.edge_index.shape == data.edge_index.shape


def test_explain_node_gnn_explainer_style_returns_dict() -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, x, edge_index):
            return torch.randn(x.size(0), 2)

    data = Data(x=torch.randn(4, 8), edge_index=torch.tensor([[0, 1], [1, 0]]))
    result = explain_node_gnn_explainer_style(DummyModel(), data, target_node=0, num_edges=2, epochs=2)
    assert isinstance(result, dict)
    assert "top_edges" in result
    assert "minimal_subgraph_node_ids" in result
