"""Tests for ml.graph.time_encoding: sinusoidal_time_encoding, time_encoding, edge_time_features."""
import pytest

pytest.importorskip("torch")
import torch

from ml.graph.time_encoding import (
    sinusoidal_time_encoding,
    time_encoding,
    edge_time_features,
)


def test_sinusoidal_time_encoding_shape() -> None:
    ts = torch.tensor([0.0, 1.0, 2.0])
    out = sinusoidal_time_encoding(ts, dim=8)
    assert out.shape == (3, 8)


def test_sinusoidal_time_encoding_1d_broadcast() -> None:
    ts = torch.tensor([1000.0])  # (1,)
    out = sinusoidal_time_encoding(ts, dim=8)
    assert out.shape == (1, 8)


def test_sinusoidal_time_encoding_odd_dim() -> None:
    ts = torch.tensor([0.0])
    out = sinusoidal_time_encoding(ts, dim=7)
    assert out.shape == (1, 7)


def test_time_encoding_list_input() -> None:
    out = time_encoding([0.0, 1.0], dim=8)
    assert out.shape == (2, 8)


def test_time_encoding_learned_returns_zeros() -> None:
    out = time_encoding([0.0], dim=8, learned=True)
    assert out.shape == (1, 8)
    assert torch.allclose(out, torch.zeros(1, 8))


def test_time_encoding_scalar_broadcast() -> None:
    out = time_encoding(12345.0, dim=8)
    assert out.shape == (1, 8)


def test_edge_time_features_shape() -> None:
    E = 5
    edge_ts = torch.rand(E)
    first_seen = torch.zeros(E)
    last_seen = torch.ones(E) * 100
    out = edge_time_features(edge_ts, first_seen, last_seen, dim=8)
    # (E, dim + 2) = (5, 10)
    assert out.shape == (E, 8 + 2)


def test_edge_time_features_recency_bounded() -> None:
    edge_ts = torch.tensor([0.0])
    first_seen = torch.tensor([0.0])
    last_seen = torch.tensor([86400.0])  # 1 day
    out = edge_time_features(edge_ts, first_seen, last_seen, dim=4)
    assert out.shape == (1, 6)
    # recency = 1/(1+1) = 0.5
    recency = out[0, -1].item()
    assert 0 <= recency <= 1
