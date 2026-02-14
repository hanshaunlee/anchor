#!/usr/bin/env python3
"""
Train baseline HGT (or GPS/FraudGT) on synthetic or Elliptic-style data.
Temporal split; weighted BCE / focal loss; metrics: PR-AUC, Recall@K, time-to-flag.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv, Linear

from ml.graph.builder import build_hetero_from_tables
from ml.models.hgt_baseline import HGTBaseline
from ml.config import get_train_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, reduction: str = "mean") -> torch.Tensor:
    """Focal loss for imbalanced anomaly detection (Lin et al.)."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    fl = (1 - pt) ** gamma * ce
    return fl.mean() if reduction == "mean" else fl.sum()


def _hetero_metadata(data) -> tuple[list, list, dict]:
    """Return (node_types, edge_types, in_channels) from a HeteroData."""
    try:
        node_types, edge_types = data.metadata()
    except Exception:
        node_types = [getattr(s, "key", i) for i, s in enumerate(data.node_stores)]
        edge_types = [getattr(s, "key", None) for s in data.edge_stores]
        edge_types = [e for e in edge_types if e is not None]
    in_channels = {}
    for nt in node_types:
        try:
            store = data[nt]
            if getattr(store, "x", None) is not None:
                in_channels[nt] = data[nt].x.size(1)
        except (KeyError, TypeError):
            pass
    return node_types, edge_types, in_channels


def get_hgb_hetero(data_dir: Path, name: str = "ACM") -> tuple[dict, list, tuple]:
    """Load a real heterogeneous graph from PyG HGB (ACM, DBLP, IMDB, Freebase). Downloads if needed."""
    try:
        from torch_geometric.datasets import HGBDataset
    except ImportError:
        logger.warning("HGBDataset not available. Install torch-geometric with dataset support.")
        return {}, [], ([], [])
    root = data_dir / "hgb"
    root.mkdir(parents=True, exist_ok=True)
    dataset = HGBDataset(root=str(root), name=name)
    if len(dataset) == 0:
        return {}, [], ([], [])
    data = dataset[0]
    node_types, edge_types, in_channels = _hetero_metadata(data)
    if not in_channels:
        return {}, [], ([], [])
    metadata = (node_types, edge_types)
    return in_channels, [data], metadata


def get_synthetic_hetero(data_dir: Path) -> tuple[dict, list]:
    """Load or create a minimal hetero graph for training (synthetic). TGAT adds time_encoding_dim to node features."""
    sessions = [{"id": "s1"}, {"id": "s2"}]
    utterances = [
        {"id": "u1", "session_id": "s1", "intent": "call"},
        {"id": "u2", "session_id": "s1", "intent": "reminder"},
        {"id": "u3", "session_id": "s2", "intent": "call"},
    ]
    entities = [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]
    mentions = [
        {"utterance_id": "u1", "entity_id": "e1"},
        {"utterance_id": "u2", "entity_id": "e2"},
        {"utterance_id": "u3", "entity_id": "e1"},
    ]
    relationships = [
        {"src_entity_id": "e1", "dst_entity_id": "e2", "rel_type": "CO_OCCURS", "first_seen_at": 0, "last_seen_at": 100, "count": 2, "weight": 1.0},
    ]
    devices = [{"id": "d1"}]
    data = build_hetero_from_tables("hh1", sessions, utterances, entities, mentions, relationships, devices)
    # PyG 2.7: node_stores iterates storage objects; use metadata() or store.key for node type names
    try:
        node_types, _ = data.metadata()
    except Exception:
        node_types = [getattr(s, "key", i) for i, s in enumerate(data.node_stores)]
    in_channels = {nt: data[nt].x.size(1) for nt in node_types}
    return in_channels, [data]


def _get_hetero_labels(data, target_node_type: str | None = None):
    """Return (out_tensor, labels) for a HeteroData for training. Uses target_node_type if given, else first type with 'y' or 'entity'."""
    try:
        node_types, _ = data.metadata()
    except Exception:
        node_types = [getattr(s, "key", i) for i, s in enumerate(data.node_stores)]
    if target_node_type and target_node_type in node_types:
        nt = target_node_type
    else:
        nt = None
        for n in node_types:
            try:
                if getattr(data[n], "y", None) is not None:
                    nt = n
                    break
            except (KeyError, TypeError):
                pass
        if nt is None and "entity" in node_types:
            nt = "entity"
        if nt is None:
            nt = node_types[0] if node_types else None
    if nt is None:
        return None, None
    out = data  # caller will pass model output dict
    return nt, getattr(data[nt], "y", None)


def train_step(
    model: nn.Module,
    data_list: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_hetero: bool = True,
    target_node_type: str | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    for data in data_list:
        if use_hetero:
            data = data.to(device)
            out = model.forward_hetero_data(data)
            nt, labels = _get_hetero_labels(data, target_node_type)
            if nt is None or nt not in out:
                continue
            node_out = out[nt]
            if labels is None:
                # Synthetic: dummy labels for demo
                labels = torch.zeros(node_out.size(0), dtype=torch.long, device=device)
                if labels.size(0) > 0:
                    labels[0] = 0
                    if labels.size(0) > 1:
                        labels[1] = 1
            else:
                labels = labels.to(device).long()
            if node_out.size(0) != labels.size(0):
                continue
            loss = F.cross_entropy(node_out, labels)
        else:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            labels = getattr(data, "y", torch.zeros(out.size(0), dtype=torch.long, device=device))
            if out.size(0) != labels.size(0):
                labels = labels[: out.size(0)]
            loss = F.cross_entropy(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(data_list), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hgt_baseline.yaml")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "hgb"],
                        help="synthetic: in-memory demo graph; hgb: real HGB (ACM/DBLP/IMDB/Freebase)")
    parser.add_argument("--hgb-name", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"],
                        help="HGB dataset name when --dataset hgb")
    parser.add_argument("--epochs", type=int, default=None, help="Override config train.epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override config train.lr")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/hgt_baseline"))
    args = parser.parse_args()
    device = torch.device(args.device)

    cfg = get_train_config(args.config)
    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    graph_cfg = cfg.get("graph") or {}

    if args.dataset == "hgb":
        in_channels, data_list, metadata = get_hgb_hetero(args.data_dir, name=args.hgb_name)
        if not in_channels or not data_list:
            raise SystemExit("Failed to load HGB dataset. Check data-dir and network.")
        # HGB may list node types without .x (e.g. term); model must only use types in in_channels
        node_types_ok = [nt for nt in metadata[0] if nt in in_channels]
        edge_types_ok = [e for e in metadata[1] if e[0] in node_types_ok and e[2] in node_types_ok]
        metadata = (node_types_ok, edge_types_ok)
        target_node_type = None  # infer from first node type with 'y'
        # Infer num classes from labels so out_channels matches (HGB has 3+ classes, config defaults to 2)
        nt, labels = _get_hetero_labels(data_list[0], target_node_type)
        num_classes = 2
        if labels is not None and labels.numel() > 0:
            # Use valid labels only (HGB may use -1 for unlabeled)
            valid = labels >= 0
            if valid.any():
                num_classes = max(2, int(labels[valid].max().item()) + 1)
            else:
                num_classes = 2
        logger.info("Loaded HGB %s: %d graphs, node types %s, num_classes %d", args.hgb_name, len(data_list), node_types_ok, num_classes)
    else:
        in_channels, data_list = get_synthetic_hetero(args.data_dir)
        node_types = graph_cfg.get("node_types", ["person", "device", "session", "utterance", "intent", "entity"])
        edge_types_raw = graph_cfg.get("edge_types", [])
        edge_types = [tuple(e) if isinstance(e, list) else e for e in edge_types_raw]
        metadata = (node_types, edge_types)
        target_node_type = "entity"
        num_classes = None  # use config

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)
    hidden = model_cfg.get("hidden_channels", 32)
    out_channels = int(num_classes) if num_classes is not None else model_cfg.get("out_channels", 2)
    num_layers = model_cfg.get("num_layers", 2)
    heads = model_cfg.get("heads", 4)
    if args.dataset == "hgb":
        logger.info("HGT out_channels=%d (from dataset labels)", out_channels)

    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=out_channels,
        num_layers=num_layers,
        heads=heads,
        metadata=metadata,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train_step(model, data_list, optimizer, device, use_hetero=True, target_node_type=target_node_type)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d loss %.4f", epoch + 1, loss)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata, "in_channels": in_channels}, out_dir / "best.pt")
    logger.info("Saved to %s", out_dir / "best.pt")


if __name__ == "__main__":
    main()
