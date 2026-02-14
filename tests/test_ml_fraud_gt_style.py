"""Tests for ml.models.fraud_gt_style: FraudGTStyle forward."""
import pytest

pytest.importorskip("torch")
import torch

from ml.models.fraud_gt_style import FraudGTStyle


def test_fraud_gt_style_forward() -> None:
    model = FraudGTStyle(
        in_channels=8,
        hidden_channels=16,
        out_channels=2,
        edge_attr_dim=4,
        num_layers=1,
    )
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    out = model(x, edge_index, edge_attr)
    assert out.shape == (5, 2)
