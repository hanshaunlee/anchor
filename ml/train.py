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


def train_step(
    model: nn.Module,
    data_list: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_hetero: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    for data in data_list:
        if use_hetero:
            data = data.to(device)
            out = model.forward_hetero_data(data)
            # Dummy labels on entity nodes for demo
            entity_out = out["entity"]
            labels = torch.zeros(entity_out.size(0), dtype=torch.long, device=device)
            if labels.size(0) > 0:
                labels[0] = 0  # normal
                if labels.size(0) > 1:
                    labels[1] = 1  # anomaly
            if entity_out.size(0) != labels.size(0):
                continue
            loss = F.cross_entropy(entity_out, labels)
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
    node_types = graph_cfg.get("node_types", ["person", "device", "session", "utterance", "intent", "entity"])
    edge_types_raw = graph_cfg.get("edge_types", [])
    edge_types = [tuple(e) if isinstance(e, list) else e for e in edge_types_raw]
    metadata = (node_types, edge_types)

    in_channels, data_list = get_synthetic_hetero(args.data_dir)
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)
    hidden = model_cfg.get("hidden_channels", 32)
    out_channels = model_cfg.get("out_channels", 2)
    num_layers = model_cfg.get("num_layers", 2)
    heads = model_cfg.get("heads", 4)

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
        loss = train_step(model, data_list, optimizer, device, use_hetero=True)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d loss %.4f", epoch + 1, loss)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata, "in_channels": in_channels}, out_dir / "best.pt")
    logger.info("Saved to %s", out_dir / "best.pt")


if __name__ == "__main__":
    main()
