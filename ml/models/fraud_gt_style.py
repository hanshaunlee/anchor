"""
FraudGT-style module: edge-attribute attention bias + edge gating (ICAIF 2024).
Adapted for Independence Graph: edge attributes = rel_type, recency, count, confidence.
Ref: FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeAttrAttentionBias(nn.Module):
    """Edge-attribute attention bias: project edge_attr to a scalar bias per head."""

    def __init__(self, edge_attr_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.lin = nn.Linear(edge_attr_dim, heads)

    def forward(self, edge_attr: Tensor) -> Tensor:
        return self.lin(edge_attr)  # (E, heads)


class EdgeGating(nn.Module):
    """Gate message by edge attributes (e.g. recency, count)."""

    def __init__(self, edge_attr_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_lin = nn.Linear(edge_attr_dim, hidden_dim)

    def forward(self, edge_attr: Tensor) -> Tensor:
        return torch.sigmoid(self.gate_lin(edge_attr))  # (E, hidden)


class FraudGTStyleConv(MessagePassing):
    """
    Single conv layer: message passing with edge-attribute attention bias and edge gating.
    alpha_ij = softmax_j( Q_i K_j^T / sqrt(d) + edge_bias_ij ) * edge_gate_ij
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: int,
        heads: int = 4,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.head_dim = out_channels // heads
        assert self.head_dim * heads == out_channels
        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)
        self.edge_bias = EdgeAttrAttentionBias(edge_attr_dim, heads)
        self.edge_gate = EdgeGating(edge_attr_dim, out_channels)
        self.lin_skip = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        n = x_i.size(0)
        q = self.lin_q(x_i).view(-1, self.heads, self.head_dim)
        k = self.lin_k(x_j).view(-1, self.heads, self.head_dim)
        v = self.lin_v(x_j).view(-1, self.heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q * k).sum(dim=-1) * scale  # (E, heads)
        bias = self.edge_bias(edge_attr)  # (E, heads)
        attn = attn + bias
        attn = softmax(attn, index, ptr, size_i)
        gate = self.edge_gate(edge_attr)  # (E, out)
        msg = (attn.unsqueeze(-1) * v).view(-1, self.out_channels) * gate
        return msg

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor], dim_size: Optional[int]) -> Tensor:
        return super().aggregate(inputs, index, ptr, dim_size)

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return aggr_out + self.lin_skip(x)


class FraudGTStyle(nn.Module):
    """
    FraudGT-style stack for homogeneous graph with edge attributes.
    Use on entity–entity subgraph (e.g. co_occurs with edge_attr) for risk scoring.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_attr_dim: int = 4,
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                FraudGTStyleConv(hidden_channels, hidden_channels, edge_attr_dim, heads=heads)
            )
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        x = self.lin_in(x).relu()
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) + x
            x = x.relu()
        return self.lin_out(x)
