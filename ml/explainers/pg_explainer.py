"""
PGExplainer-style: parameterized explainer that learns a shared explanation network
for efficiency (Luo et al., NeurIPS 2020). Faster than per-instance GNNExplainer.
Ref: Parameterized Explainer for Graph Neural Network.
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class PGExplainerStyle(nn.Module):
    """
    Learns edge importance with a small MLP over (h_i || h_j || edge_attr) for each edge.
    Shared across instances; trained with reparameterization (Gumbel) or direct sigmoid.
    """

    def __init__(self, hidden_dim: int, edge_attr_dim: int = 0):
        super().__init__()
        input_dim = hidden_dim * 2 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src = node_emb[src]
        h_dst = node_emb[dst]
        if edge_attr is not None:
            inp = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        else:
            inp = torch.cat([h_src, h_dst], dim=-1)
        return self.mlp(inp).squeeze(-1)

    def edge_weights(self, node_emb: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(node_emb, edge_index, edge_attr))


def explain_with_pg(
    pg_explainer: PGExplainerStyle,
    node_emb: torch.Tensor,
    data: Data,
    top_k: int = 20,
) -> dict:
    """Produce explanation_json from PGExplainer edge weights."""
    if data.edge_attr is not None:
        w = pg_explainer.edge_weights(node_emb, data.edge_index, data.edge_attr)
    else:
        w = pg_explainer.edge_weights(node_emb, data.edge_index, None)
    w = w.detach()
    edge_index = data.edge_index
    ranked = []
    for e in range(edge_index.size(1)):
        ranked.append({
            "src": int(edge_index[0, e]),
            "dst": int(edge_index[1, e]),
            "score": float(w[e]),
        })
    ranked.sort(key=lambda x: -x["score"])
    top_edges = ranked[:top_k]
    threshold = 0.5
    kept = [(int(edge_index[0, e].item()), int(edge_index[1, e].item())) for e in range(edge_index.size(1)) if w[e].item() > threshold]
    node_set = set()
    for a, b in kept:
        node_set.add(a)
        node_set.add(b)
    return {
        "top_edges": top_edges,
        "minimal_subgraph_node_ids": list(node_set),
        "minimal_subgraph_edges": kept,
        "summary": f"PGExplainer: {len(node_set)} nodes, {len(kept)} edges.",
        "method": "pg_explainer",
    }
