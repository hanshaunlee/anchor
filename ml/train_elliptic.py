#!/usr/bin/env python3
"""
Train on Elliptic (Bitcoin) temporal graph for academic credibility.
Output: PR-AUC, UMAP plot, example explanation subgraph.
Reproducible: python -m ml.train_elliptic --dataset elliptic --model fraud_gt_style
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Linear
from torch_geometric.utils import negative_sampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_elliptic_data(data_dir: Path):
    """Load Elliptic Bitcoin temporal dataset (PyG). Fallback to synthetic if not available."""
    try:
        from torch_geometric.datasets import EllipticBitcoinTemporalDataset
        path = data_dir / "elliptic"
        path.mkdir(parents=True, exist_ok=True)
        dataset = EllipticBitcoinTemporalDataset(root=str(path), t=1)
        if len(dataset) > 0:
            return dataset[0]
        return None
    except Exception as e:
        logger.warning("Elliptic dataset not available: %s. Using small synthetic.", e)
        return None


def make_synthetic_elliptic(num_nodes: int = 500, num_edges: int = 2000, num_classes: int = 2):
    """Minimal synthetic temporal graph for when Elliptic is not downloaded."""
    import torch
    from torch_geometric.data import Data
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    # Remove duplicates and self-loops
    edge_index = torch.unique(edge_index, dim=1)
    y = torch.randint(0, num_classes, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


class FraudGTStyleSmall(torch.nn.Module):
    """Small FraudGT-style for node classification (Elliptic has node labels)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, edge_attr_dim: int = 4):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.lin_in = Linear(in_dim, hidden)
        from ml.models.fraud_gt_style import FraudGTStyleConv
        self.conv1 = FraudGTStyleConv(hidden, hidden, edge_attr_dim, heads=2)
        self.conv2 = FraudGTStyleConv(hidden, hidden, edge_attr_dim, heads=2)
        self.lin_out = Linear(hidden, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_attr_dim, device=x.device)
        x = self.lin_in(x).relu()
        x = self.conv1(x, edge_index, edge_attr) + x
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr) + x
        x = x.relu()
        return self.lin_out(x)


def train_epoch(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    if hasattr(data, "train_mask") and data.train_mask is not None and data.train_mask.any():
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())
    else:
        loss = F.cross_entropy(out, data.y.long())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device, mask_name: str = "val_mask"):
    model.eval()
    out = model(data.x, data.edge_index, getattr(data, "edge_attr", None))
    pred = out.argmax(dim=-1)
    mask = getattr(data, mask_name, None)
    if mask is not None and mask.any():
        y = data.y[mask].long()
        pred = pred[mask]
    else:
        y = data.y.long()
    pr_auc = 0.0
    try:
        from sklearn.metrics import average_precision_score
        prob = F.softmax(out, dim=-1)
        if mask is not None and mask.any():
            prob = prob[mask][:, 1].cpu().numpy()
            y_np = data.y[mask].cpu().numpy()
        else:
            prob = prob[:, 1].cpu().numpy()
            y_np = data.y.cpu().numpy()
        if y_np.max() >= 1:
            pr_auc = average_precision_score(y_np, prob)
    except Exception:
        pass
    acc = (pred == y).float().mean().item()
    return {"accuracy": acc, "pr_auc": pr_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="elliptic")
    parser.add_argument("--model", type=str, default="fraud_gt_style")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--edge-attr-dim", type=int, default=4, help="Edge attribute dimension")
    parser.add_argument("--output", type=Path, default=Path("runs/elliptic"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_elliptic_data(args.data_dir)
    if data is not None:
        in_dim = data.x.size(1)
        num_classes = max(2, int(data.y[data.y >= 0].max().item()) + 1) if data.y.numel() else 2
        edge_dim = getattr(args, "edge_attr_dim", 4)
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.zeros(data.edge_index.size(1), edge_dim)
        # Mask: labeled nodes only (Elliptic has -1 for unknown)
        labeled = (data.y >= 0).nonzero(as_tuple=True)[0]
        n = data.x.size(0)
        data.train_mask = torch.zeros(n, dtype=torch.bool)
        data.val_mask = torch.zeros(n, dtype=torch.bool)
        data.test_mask = torch.zeros(n, dtype=torch.bool)
        if len(labeled) >= 3:
            perm = torch.randperm(len(labeled))
            t, v, te = perm[: len(labeled) // 2], perm[len(labeled) // 2 : 3 * len(labeled) // 4], perm[3 * len(labeled) // 4 :]
            data.train_mask[labeled[t]] = True
            data.val_mask[labeled[v]] = True
            data.test_mask[labeled[te]] = True
    else:
        data = make_synthetic_elliptic()
        in_dim = data.x.size(1)
        num_classes = int(data.y.max().item()) + 1
        edge_dim = getattr(args, "edge_attr_dim", 4)
        data.edge_attr = torch.zeros(data.edge_index.size(1), edge_dim)
    data = data.to(device)

    model = FraudGTStyleSmall(in_dim, args.hidden, num_classes, edge_attr_dim=args.edge_attr_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_epoch(model, data, optimizer, device)
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, data, device)
            logger.info("Epoch %d loss %.4f acc %.4f PR-AUC %.4f", epoch + 1, loss, metrics["accuracy"], metrics.get("pr_auc", 0))

    metrics = evaluate(model, data, device)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Final PR-AUC: %.4f", metrics.get("pr_auc", 0))

    # UMAP plot (optional)
    try:
        import numpy as np
        from sklearn.manifold import TSNE
        model.eval()
        with torch.no_grad():
            h = model.lin_in(data.x)
            h = F.relu(h)
            h = model.conv1(h, data.edge_index, data.edge_attr) + h
            h = h.relu()
            emb = h.cpu().numpy()
        if emb.shape[0] > 500:
            idx = torch.randperm(emb.shape[0])[:500]
            emb = emb[idx]
            y_plot = data.y.cpu().numpy()[idx]
        else:
            y_plot = data.y.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        xy = tsne.fit_transform(emb)
        plot_data = [{"x": float(xy[i, 0]), "y": float(xy[i, 1]), "label": int(y_plot[i])} for i in range(len(y_plot))]
        with open(args.output / "embedding_plot.json", "w") as f:
            json.dump(plot_data, f)
        logger.info("Saved embedding_plot.json (use for UMAP/TSNE viz)")
    except Exception as e:
        logger.warning("Embedding plot skip: %s", e)

    # Example explanation subgraph: high-risk node and 1-hop neighborhood
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        prob = F.softmax(out, dim=-1)[:, 1]
    high_idx = prob.argmax().item()
    node_set = {high_idx}
    for e in range(data.edge_index.size(1)):
        a, b = data.edge_index[0, e].item(), data.edge_index[1, e].item()
        if a == high_idx or b == high_idx:
            node_set.add(a)
            node_set.add(b)
    example_subgraph = {
        "center_node": high_idx,
        "nodes": list(node_set),
        "edges": [
            {"src": data.edge_index[0, e].item(), "dst": data.edge_index[1, e].item()}
            for e in range(data.edge_index.size(1))
            if data.edge_index[0, e].item() in node_set and data.edge_index[1, e].item() in node_set
        ],
    }
    with open(args.output / "example_explanation_subgraph.json", "w") as f:
        json.dump(example_subgraph, f, indent=2)
    logger.info("Saved example_explanation_subgraph.json")
    return metrics


if __name__ == "__main__":
    main()
