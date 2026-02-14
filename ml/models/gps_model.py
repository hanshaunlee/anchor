"""
GraphGPS-style model: local message passing + global attention (Rampasek et al., NeurIPS 2022).
Recipe for a General, Powerful, Scalable Graph Transformer; linear complexity.
Ref: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GPSConv.html
We use a homogeneous projection of the hetero graph (entity/session nodes) for risk scoring.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GPSConv, Linear
from torch_geometric.data import Data


class GPSRiskModel(nn.Module):
    """
    GPS (local MP + global attention) on a flattened graph for session/entity risk.
    Input: homogeneous Data with node_type one-hot + features; edge_index; edge_attr for temporal.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        self.lin_in = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GPSConv(
                    hidden_channels,
                    conv=GATConv(hidden_channels, hidden_channels // heads, heads=heads, add_self_loops=False),
                    heads=heads,
                    dropout=dropout,
                )
            )
        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.lin_in(x).relu()
        for conv in self.convs:
            x = conv(x, edge_index) + x
            x = x.relu()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x)
