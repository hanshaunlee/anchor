"""Tests for ml.models: HGTBaseline, GPSRiskModel forward shapes."""
import pytest

pytest.importorskip("torch")
import torch
from torch_geometric.data import HeteroData

from ml.models.hgt_baseline import HGTBaseline
from ml.models.gps_model import GPSRiskModel


def test_hgt_baseline_forward_dict() -> None:
    in_channels = {"person": 8, "device": 8, "session": 8, "utterance": 16, "intent": 8, "entity": 16}
    metadata = (
        ["person", "device", "session", "utterance", "intent", "entity"],
        [
            ("person", "uses", "device"),
            ("session", "has", "utterance"),
            ("utterance", "expresses", "intent"),
            ("utterance", "mentions", "entity"),
            ("entity", "co_occurs", "entity"),
        ],
    )
    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=2,
        num_layers=1,
        heads=2,
        metadata=metadata,
    )
    x_dict = {
        nt: torch.randn(3 if nt == "entity" else 2, in_channels[nt])
        for nt in metadata[0]
    }
    edge_index_dict = {
        ("entity", "co_occurs", "entity"): torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    }
    out = model.forward(x_dict, edge_index_dict)
    assert "entity" in out
    assert out["entity"].shape == (3, 2)


def test_hgt_baseline_forward_hetero_data() -> None:
    data = HeteroData()
    data["person"].x = torch.randn(1, 8)
    data["device"].x = torch.randn(2, 8)
    data["session"].x = torch.randn(2, 8)
    data["utterance"].x = torch.randn(3, 16)
    data["intent"].x = torch.randn(2, 8)
    data["entity"].x = torch.randn(4, 16)
    data["person", "uses", "device"].edge_index = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
    data["entity", "co_occurs", "entity"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    in_channels = {"person": 8, "device": 8, "session": 8, "utterance": 16, "intent": 8, "entity": 16}
    metadata = (
        list(in_channels.keys()),
        [("person", "uses", "device"), ("entity", "co_occurs", "entity")],
    )
    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=2,
        num_layers=1,
        heads=2,
        metadata=metadata,
    )
    out = model.forward_hetero_data(data)
    assert out["entity"].shape == (4, 2)


def test_gps_model_forward() -> None:
    model = GPSRiskModel(
        in_channels=16,
        hidden_channels=32,
        out_channels=2,
        num_layers=1,
        heads=2,
    )
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (5, 2)
