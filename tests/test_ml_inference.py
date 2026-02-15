"""Tests for ml.inference: load_model, run_inference."""
import json
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from ml.models.hgt_baseline import HGTBaseline
from ml.inference import load_model, run_inference


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    in_channels = {"person": 8, "device": 8, "session": 8, "utterance": 16, "intent": 8, "entity": 16}
    metadata = (
        list(in_channels.keys()),
        [("person", "uses", "device"), ("entity", "co_occurs", "entity")],
    )
    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=2,
        num_layers=2,
        heads=4,
        metadata=metadata,
    )
    ckpt = {
        "in_channels": in_channels,
        "metadata": metadata,
        "model_state": model.state_dict(),
    }
    p = tmp_path / "fake.pt"
    torch.save(ckpt, p)
    return p


def test_load_model(fake_checkpoint: Path) -> None:
    device = torch.device("cpu")
    model, target_node_type = load_model(fake_checkpoint, device)
    assert isinstance(model, HGTBaseline)
    assert model.out_channels == 2
    assert target_node_type is None  # old checkpoint has no target_node_type


def test_run_inference_returns_risk_list(fake_checkpoint: Path) -> None:
    from ml.train import get_synthetic_hetero
    in_channels, data_list = get_synthetic_hetero(Path("data/synthetic"))
    data = data_list[0]
    model, _ = load_model(fake_checkpoint, torch.device("cpu"))
    risk_list, expl = run_inference(model, data, torch.device("cpu"), explain_node_idx=None)
    assert isinstance(risk_list, list)
    assert len(risk_list) == data["entity"].num_nodes
    for r in risk_list:
        assert r["node_type"] == "entity"
        assert "node_index" in r
        assert "score" in r
        assert "label" in r
        # Real incident embedding (model pooled representation) when return_embeddings=True
        assert "embedding" in r
        assert isinstance(r["embedding"], list)
        assert len(r["embedding"]) in (32, 128)  # hidden_channels from checkpoint (32 legacy, 128 current)
    assert expl is None


def test_run_inference_with_explain_node(fake_checkpoint: Path) -> None:
    from ml.train import get_synthetic_hetero
    in_channels, data_list = get_synthetic_hetero(Path("data/synthetic"))
    data = data_list[0]
    model, _ = load_model(fake_checkpoint, torch.device("cpu"))
    risk_list, expl = run_inference(model, data, torch.device("cpu"), explain_node_idx=0)
    assert expl is None or isinstance(expl, dict)
