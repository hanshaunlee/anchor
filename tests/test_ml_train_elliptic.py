"""Tests for ml.train_elliptic: make_synthetic_elliptic, FraudGTStyleSmall, train_epoch, evaluate, get_elliptic_data."""
import pytest
from pathlib import Path

pytest.importorskip("torch")
import torch

from ml.train_elliptic import (
    make_synthetic_elliptic,
    FraudGTStyleSmall,
    train_epoch,
    evaluate,
    get_elliptic_data,
)


def test_make_synthetic_elliptic_default() -> None:
    data = make_synthetic_elliptic()
    assert data.x.shape[0] == 500
    assert data.x.shape[1] == 8
    assert data.edge_index.shape[0] == 2
    assert data.y.shape[0] == 500
    assert data.y.dtype in (torch.long, torch.int64) or data.y.dtype == torch.int32


def test_make_synthetic_elliptic_custom() -> None:
    data = make_synthetic_elliptic(num_nodes=20, num_edges=50, num_classes=3)
    assert data.x.shape == (20, 8)
    assert data.edge_index.size(1) <= 50
    assert data.y.max().item() < 3


def test_fraud_gt_style_small_forward() -> None:
    model = FraudGTStyleSmall(in_dim=8, hidden=16, out_dim=2, edge_attr_dim=4)
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.randn(3, 4)
    out = model(x, edge_index, edge_attr)
    assert out.shape == (5, 2)


def test_fraud_gt_style_small_forward_no_edge_attr() -> None:
    model = FraudGTStyleSmall(in_dim=8, hidden=16, out_dim=2, edge_attr_dim=4)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    out = model(x, edge_index, edge_attr=None)
    assert out.shape == (4, 2)


def test_train_epoch() -> None:
    data = make_synthetic_elliptic(num_nodes=30, num_edges=80)
    num_edges = data.edge_index.size(1)
    data.edge_attr = torch.randn(num_edges, 4)
    model = FraudGTStyleSmall(8, 16, 2, edge_attr_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_epoch(model, data, optimizer, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss >= 0


def test_evaluate() -> None:
    data = make_synthetic_elliptic(num_nodes=25, num_edges=60)
    model = FraudGTStyleSmall(8, 16, 2, edge_attr_dim=4)
    metrics = evaluate(model, data, torch.device("cpu"))
    assert "accuracy" in metrics
    assert "pr_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["pr_auc"] >= 0


def test_evaluate_with_val_mask() -> None:
    data = make_synthetic_elliptic(num_nodes=20, num_edges=40)
    data.val_mask = torch.ones(20, dtype=torch.bool)
    data.val_mask[:5] = False
    model = FraudGTStyleSmall(8, 16, 2, edge_attr_dim=4)
    metrics = evaluate(model, data, torch.device("cpu"), mask_name="val_mask")
    assert "accuracy" in metrics


def test_get_elliptic_data_returns_none_or_data(tmp_path: Path) -> None:
    out = get_elliptic_data(tmp_path)
    assert out is None or (hasattr(out, "x") and hasattr(out, "edge_index"))
