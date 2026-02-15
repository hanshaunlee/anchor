"""Tests for ml.train: focal_loss, _hetero_metadata, get_synthetic_hetero, _get_hetero_labels, train_step."""
import pytest
from pathlib import Path

pytest.importorskip("torch")
import torch
from torch_geometric.data import HeteroData

from ml.train import (
    focal_loss,
    _hetero_metadata,
    get_synthetic_hetero,
    _get_hetero_labels,
    train_step,
)
from ml.models.hgt_baseline import HGTBaseline


def test_focal_loss_shape() -> None:
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loss = focal_loss(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_focal_loss_reduction_sum() -> None:
    logits = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    loss = focal_loss(logits, targets, reduction="sum")
    assert loss.dim() == 0


def test_hetero_metadata() -> None:
    data = HeteroData()
    data["entity"].x = torch.randn(3, 8)
    data["entity"].num_nodes = 3
    data["entity", "co_occurs", "entity"].edge_index = torch.tensor([[0, 1], [1, 0]])
    node_types, edge_types, in_channels = _hetero_metadata(data)
    assert "entity" in node_types or len(node_types) >= 1
    assert "entity" in in_channels
    assert in_channels["entity"] == 8


def test_get_synthetic_hetero() -> None:
    in_channels, data_list = get_synthetic_hetero(Path("data/synthetic"))
    assert isinstance(in_channels, dict)
    assert len(data_list) == 1
    assert data_list[0]["entity"].num_nodes == 3


def test_get_hetero_labels_entity_default() -> None:
    data = HeteroData()
    data["entity"].x = torch.randn(3, 8)
    data["entity"].y = torch.tensor([0, 1, 0])
    nt, labels = _get_hetero_labels(data)
    assert nt == "entity"
    assert labels is not None
    assert labels.size(0) == 3


def test_get_hetero_labels_no_y_returns_entity() -> None:
    in_channels, data_list = get_synthetic_hetero(Path("data/synthetic"))
    data = data_list[0]
    nt, labels = _get_hetero_labels(data)
    assert nt == "entity"
    assert labels is None


def test_train_step_synthetic_hetero() -> None:
    in_channels, data_list, _synthetic_config, _graph_stats, _entity_ids = get_synthetic_hetero(Path("data/synthetic"), seed=42)
    data = data_list[0]
    try:
        node_types, edge_types = data.metadata()
    except Exception:
        node_types = list(in_channels.keys())
        edge_types = []
    metadata = (node_types, edge_types)
    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=16,
        out_channels=2,
        num_layers=1,
        heads=2,
        metadata=metadata,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_loss, ce_loss = train_step(model, data_list, optimizer, torch.device("cpu"), use_hetero=True, target_node_type="entity")
    assert isinstance(ce_loss, float)
    assert ce_loss >= 0
