"""
GNNExplainer-style instance explanation: learn edge mask + feature mask to maximize
mutual information between model prediction and subgraph (Ying et al., NeurIPS 2019).
Ref: GNNExplainer: Generating Explanations for Graph Neural Networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


class GNNExplainerStyle(nn.Module):
    """
    Learns a soft edge mask (and optional feature mask) for one instance (node or graph).
    Objective: maximize MI between masked subgraph and model output; minimize mask entropy.
    """

    def __init__(self, num_edges: int, num_features: int = 0, allow_edge_mask: bool = True):
        super().__init__()
        self.allow_edge_mask = allow_edge_mask
        if allow_edge_mask:
            self.edge_mask = nn.Parameter(torch.ones(num_edges) * 0.5)
        if num_features > 0:
            self.feature_mask = nn.Parameter(torch.ones(num_features) * 0.5)
        else:
            self.feature_mask = None

    def get_edge_mask(self) -> torch.Tensor:
        if not self.allow_edge_mask:
            return torch.ones(self.edge_mask.size(0), device=self.edge_mask.device)
        return torch.sigmoid(self.edge_mask)

    def get_feature_mask(self) -> torch.Tensor | None:
        if self.feature_mask is None:
            return None
        return torch.sigmoid(self.feature_mask)

    def apply_masks(
        self,
        data: Data,
        edge_mask: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
    ) -> Data:
        if edge_mask is None:
            edge_mask = self.get_edge_mask()
        out = Data(x=data.x.clone(), edge_index=data.edge_index.clone())
        if data.edge_attr is not None:
            out.edge_attr = data.edge_attr * edge_mask.unsqueeze(-1)
        else:
            out.edge_attr = edge_mask.unsqueeze(-1)
        if feature_mask is not None and out.x is not None:
            out.x = out.x * feature_mask.unsqueeze(0)
        return out


def explain_node_gnn_explainer_style(
    model: nn.Module,
    data: Data,
    target_node: int,
    num_edges: int,
    epochs: int = 200,
    lr: float = 0.01,
) -> dict:
    """
    Optimize edge mask for one node's prediction. Returns ranked edges and minimal subgraph.
    """
    explainer = GNNExplainerStyle(num_edges=num_edges, num_features=data.x.size(1) if data.x is not None else 0)
    opt = torch.optim.Adam(explainer.parameters(), lr=lr)
    model.eval()
    for _ in range(epochs):
        opt.zero_grad()
        mask = explainer.get_edge_mask()
        masked_data = explainer.apply_masks(data, edge_mask=mask)
        logits = model(masked_data.x, masked_data.edge_index)
        if logits.dim() == 2:
            pred = logits[target_node : target_node + 1]
        else:
            pred = logits
        # MI-style: encourage pred to stay confident; regularize mask
        loss = -pred.log_softmax(dim=-1).max() + 0.01 * (mask * (1 - mask)).sum()
        loss.backward()
        opt.step()
    edge_mask = explainer.get_edge_mask().detach()
    # Build explanation_json
    edge_index = data.edge_index
    ranked_edges = []
    for e in range(edge_index.size(1)):
        ranked_edges.append({
            "src": int(edge_index[0, e]),
            "dst": int(edge_index[1, e]),
            "score": float(edge_mask[e]),
        })
    ranked_edges.sort(key=lambda x: -x["score"])
    top_edges = ranked_edges[:20]
    threshold = 0.5
    kept_edges = [(int(edge_index[0, e].item()), int(edge_index[1, e].item())) for e in range(edge_index.size(1)) if edge_mask[e].item() > threshold]
    node_set = set()
    for a, b in kept_edges:
        node_set.add(a)
        node_set.add(b)
    if target_node not in node_set:
        node_set.add(target_node)
    return {
        "top_edges": top_edges,
        "minimal_subgraph_node_ids": list(node_set),
        "minimal_subgraph_edges": kept_edges,
        "summary": f"GNNExplainer: {len(node_set)} nodes, {len(kept_edges)} edges above threshold.",
        "method": "gnn_explainer",
    }
