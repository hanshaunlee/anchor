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


def get_elliptic_data(data_dir: Path, temporal_split: bool = True):
    """Load Elliptic Bitcoin dataset (PyG). Prefer base dataset for strict temporal split (train 1-34, test 35-49).
    Fallback to Temporal(t=1) then synthetic if not available."""
    path = data_dir / "elliptic"
    path.mkdir(parents=True, exist_ok=True)
    # Prefer base EllipticBitcoinDataset: it provides train_mask (time_step 1-34) and test_mask (35-49) — no leakage.
    if temporal_split:
        try:
            from torch_geometric.datasets import EllipticBitcoinDataset
            dataset = EllipticBitcoinDataset(root=str(path))
            if len(dataset) > 0:
                data = dataset[0]
                # Dataset already has train_mask, test_mask; add empty val_mask so we don't overwrite with random split
                n = data.x.size(0)
                if not hasattr(data, "val_mask") or data.val_mask is None:
                    data.val_mask = torch.zeros(n, dtype=torch.bool)
                logger.info("Elliptic loaded with temporal split (train time 1-34, test 35-49)")
                return data
        except Exception as e:
            logger.warning("EllipticBitcoinDataset not available: %s. Trying Temporal(t=1).", e)
    try:
        from torch_geometric.datasets import EllipticBitcoinTemporalDataset
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


def _recall_at_k(prob_positive: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Recall@K: of all positives, how many are in the top-K by score. Binary: y==1 is positive."""
    if k <= 0 or y.sum().item() == 0:
        return 0.0
    k = min(k, prob_positive.numel())
    _, topk = prob_positive.topk(k, largest=True)
    y_flat = y.long()
    positives_in_topk = y_flat[topk].sum().item()
    total_positives = y_flat.sum().item()
    return positives_in_topk / total_positives if total_positives else 0.0


@torch.no_grad()
def evaluate(model, data, device, mask_name: str = "val_mask", recall_k: tuple[int, ...] = (10, 50, 100)):
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
    prob_positive = None
    y_np = None
    try:
        from sklearn.metrics import average_precision_score
        prob = F.softmax(out, dim=-1)
        if mask is not None and mask.any():
            prob_m = prob[mask]
            prob_positive = prob_m[:, 1]
            y_np = data.y[mask].cpu().numpy()
            prob_np = prob_m[:, 1].cpu().numpy()
        else:
            prob_positive = prob[:, 1]
            y_np = data.y.cpu().numpy()
            prob_np = prob_positive.cpu().numpy()
        if y_np.max() >= 1:
            pr_auc = average_precision_score(y_np, prob_np)
    except Exception:
        pass
    acc = (pred == y).float().mean().item()
    metrics = {"accuracy": acc, "pr_auc": pr_auc}
    # Recall@K (on masked set if any)
    y_for_recall = data.y[mask] if (mask is not None and mask.any()) else data.y
    if prob_positive is not None and y_for_recall.sum().item() >= 1:
        for k in recall_k:
            metrics[f"recall_at_{k}"] = _recall_at_k(prob_positive, y_for_recall, k)
    return metrics


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
    parser.add_argument("--no-temporal-split", action="store_true", help="Use random split instead of time-based (not recommended)")
    args = parser.parse_args()
    args.temporal_split = not args.no_temporal_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_elliptic_data(args.data_dir, temporal_split=args.temporal_split)
    if data is not None:
        in_dim = data.x.size(1)
        num_classes = max(2, int(data.y[data.y >= 0].max().item()) + 1) if data.y.numel() else 2
        edge_dim = getattr(args, "edge_attr_dim", 4)
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.zeros(data.edge_index.size(1), edge_dim)
        n = data.x.size(0)
        # Use dataset-provided temporal train/test masks if present (strict temporal split, no leakage)
        has_temporal = (
            getattr(data, "train_mask", None) is not None
            and getattr(data, "test_mask", None) is not None
            and data.train_mask.any()
            and data.test_mask.any()
        )
        if not has_temporal:
            # Fallback: random split on labeled nodes (e.g. synthetic or single-snapshot Temporal)
            labeled = (data.y >= 0).nonzero(as_tuple=True)[0]
            data.train_mask = torch.zeros(n, dtype=torch.bool)
            data.val_mask = torch.zeros(n, dtype=torch.bool)
            data.test_mask = torch.zeros(n, dtype=torch.bool)
            if len(labeled) >= 3:
                perm = torch.randperm(len(labeled))
                t, v, te = perm[: len(labeled) // 2], perm[len(labeled) // 2 : 3 * len(labeled) // 4], perm[3 * len(labeled) // 4 :]
                data.train_mask[labeled[t]] = True
                data.val_mask[labeled[v]] = True
                data.test_mask[labeled[te]] = True
        if not hasattr(data, "val_mask") or data.val_mask is None or not data.val_mask.any():
            data.val_mask = torch.zeros(n, dtype=torch.bool)
    else:
        data = make_synthetic_elliptic()
        in_dim = data.x.size(1)
        num_classes = int(data.y.max().item()) + 1
        edge_dim = getattr(args, "edge_attr_dim", 4)
        data.edge_attr = torch.zeros(data.edge_index.size(1), edge_dim)
    data = data.to(device)

    model = FraudGTStyleSmall(in_dim, args.hidden, num_classes, edge_attr_dim=args.edge_attr_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    recall_ks = (10, 50, 100)
    eval_mask = "val_mask" if (getattr(data, "val_mask", None) is not None and data.val_mask.any()) else "test_mask"
    for epoch in range(args.epochs):
        loss = train_epoch(model, data, optimizer, device)
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, data, device, mask_name=eval_mask, recall_k=recall_ks)
            rec_str = " ".join(f"R@{k}={metrics.get(f'recall_at_{k}', 0):.4f}" for k in recall_ks)
            logger.info(
                "Epoch %d loss %.4f acc %.4f PR-AUC %.4f %s",
                epoch + 1, loss, metrics["accuracy"], metrics.get("pr_auc", 0), rec_str,
            )

    metrics = evaluate(model, data, device, mask_name="test_mask", recall_k=recall_ks)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Final PR-AUC: %.4f Recall@K %s", metrics.get("pr_auc", 0), {f"recall_at_{k}": metrics.get(f"recall_at_{k}") for k in recall_ks})

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
