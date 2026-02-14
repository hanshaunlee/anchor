"""
TGAT-style time encoding (Xu et al., ICLR 2020).
Inject into node features and/or edge attributes for temporal message passing.
Ref: Inductive Representation Learning on Temporal Graphs.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def sinusoidal_time_encoding(ts: Tensor, dim: int, base: float = 10000.0) -> Tensor:
    """
    Sinusoidal positional encoding over time.
    ts: (N,) or (N, 1) float timestamps (e.g. Unix time or normalized).
    Returns (N, dim).
    """
    if ts.dim() == 1:
        ts = ts.unsqueeze(1)
    half = dim // 2
    div = torch.exp(torch.arange(half, device=ts.device, dtype=ts.dtype) * (-math.log(base) / half))
    t = ts * div  # (N, half)
    enc = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
    if dim % 2:
        enc = torch.cat([enc, torch.zeros(enc.size(0), 1, device=enc.device)], dim=-1)
    return enc


def time_encoding(
    ts: Tensor | list[float],
    dim: int,
    device: Optional[torch.device] = None,
    learned: bool = False,
) -> Tensor:
    """
    Time encoding for use in models.
    ts: scalar per node/edge (will be broadcast to (N,)).
    If learned=False, uses sinusoidal. If learned=True, returns zeros (caller uses learned embedding).
    """
    if learned:
        return torch.zeros(1, dim, device=device or (ts.device if isinstance(ts, Tensor) else None))
    if not isinstance(ts, Tensor):
        ts = torch.tensor(ts, dtype=torch.float32, device=device)
    if ts.dim() == 0:
        ts = ts.unsqueeze(0)
    return sinusoidal_time_encoding(ts, dim)


def edge_time_features(
    edge_ts: Tensor,
    first_seen: Tensor,
    last_seen: Tensor,
    dim: int = 8,
) -> Tensor:
    """
    Edge temporal features: [sin_enc(edge_ts), delta_t, recency].
    edge_ts: (E,) time of edge; first_seen, last_seen: (E,) from relationship.
    Returns (E, dim + 2): time_enc (dim) + delta_t_norm + recency.
    """
    enc = sinusoidal_time_encoding(edge_ts, dim)  # (E, dim)
    delta = (last_seen - first_seen).clamp(min=0).unsqueeze(1)  # (E, 1)
    # recency: 1 / (1 + delta_days)
    recency = 1.0 / (1.0 + delta / 86400.0)
    return torch.cat([enc, delta, recency], dim=-1)
