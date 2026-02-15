#!/usr/bin/env python3
"""
Train baseline HGT (or GPS/FraudGT) on synthetic or Elliptic-style data.
Temporal split; weighted BCE / focal loss; metrics: PR-AUC, Recall@K, time-to-flag.
Reproducible: --run-dir runs/<timestamp>_<gitsha>, --seed 42; writes history.jsonl, preds, embeddings, motifs, explanations.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv, Linear

from ml.graph.builder import build_hetero_from_tables
from ml.models.hgt_baseline import HGTBaseline
from ml.config import get_train_config
from ml.run_utils import set_seed, get_default_run_dir, setup_run_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _json_serialize(obj):
    """Convert numpy/types for JSON dump."""
    if isinstance(obj, dict):
        return {k: _json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serialize(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


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


def get_supabase_export_hetero(data_dir: Path, seed: int = 42) -> tuple[dict, list, dict, dict, list[str]]:
    """Load a graph exported by scripts/export_supabase_for_gnn.py. Returns (in_channels, data_list, export_config, graph_stats, entity_ids)."""
    data_dir = Path(data_dir)
    graph_path = data_dir / "graph.pt"
    meta_path = data_dir / "meta.json"
    if not graph_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Supabase export not found: need {graph_path} and {meta_path}. "
            "Run: python scripts/export_supabase_for_gnn.py --household-id <uuid> --output <data_dir>"
        )
    # graph.pt is our own HeteroData export (PyG); weights_only=True rejects PyG types
    data = torch.load(graph_path, weights_only=False)
    with open(meta_path) as f:
        meta = json.load(f)
    in_channels = meta["in_channels"]
    entity_ids = meta.get("entity_ids") or []
    node_types = meta.get("node_types")
    edge_types = meta.get("edge_types")
    if not node_types or not edge_types:
        try:
            node_types, edge_types = data.metadata()
        except Exception:
            node_types = list(in_channels.keys())
            edge_types = []
    if isinstance(edge_types[0], list):
        edge_types = [tuple(e) for e in edge_types]
    export_config = {
        "source": "supabase_export",
        "household_id": meta.get("household_id"),
        "time_window_days": meta.get("time_window_days"),
        "num_events": meta.get("num_events"),
        "num_sessions": meta.get("num_sessions"),
        "seed": seed,
    }
    graph_stats_path = data_dir / "graph_stats.json"
    graph_stats = {}
    if graph_stats_path.exists():
        with open(graph_stats_path) as f:
            graph_stats = json.load(f)
    return in_channels, [data], export_config, graph_stats, entity_ids


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


def get_synthetic_hetero(data_dir: Path, seed: int = 42) -> tuple[dict, list, dict, dict, list[str]]:
    """Load or create a minimal hetero graph for training (synthetic).
    Returns (in_channels, data_list, synthetic_config, graph_stats, entity_ids).
    """
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
    # Same schema as inference: pipeline/agents pass devices=[] when no devices; keep training aligned.
    devices: list[dict] = []
    data = build_hetero_from_tables("hh1", sessions, utterances, entities, mentions, relationships, devices)
    try:
        node_types, edge_types = data.metadata()
    except Exception:
        node_types = [getattr(s, "key", i) for i, s in enumerate(data.node_stores)]
        edge_types = [getattr(s, "key", None) for s in data.edge_stores]
        edge_types = [e for e in edge_types if e is not None]
    in_channels = {nt: data[nt].x.size(1) for nt in node_types}
    entity_ids = [e["id"] for e in entities]
    synthetic_config = {
        "counts": {"entities": len(entities), "sessions": len(sessions), "events": len(utterances), "utterances": len(utterances), "mentions": len(mentions), "relationships": len(relationships)},
        "fraud_ratio": None,
        "seed": seed,
    }
    _add_synthetic_masks_and_labels(data, "entity", seed=seed)
    if getattr(data["entity"], "y", None) is not None and data["entity"].x.size(0):
        synthetic_config["fraud_ratio"] = float(np.mean(data["entity"].y.numpy() == 1))
    graph_stats = _graph_stats(data, node_types, edge_types)
    return in_channels, [data], synthetic_config, graph_stats, entity_ids


def get_structured_synthetic_hetero(data_dir: Path, seed: int = 42) -> tuple[dict, list, dict, dict, list[str]]:
    """Generate structured synthetic heterograph (fraud from structure: cross-session, motifs, burstiness, device sharing).
    Returns (in_channels, data_list, synthetic_config, graph_stats, entity_ids). Validates separability before return."""
    from ml.synthetic.structured_generator import generate_structured_heterograph, StructuredGeneratorConfig
    config = StructuredGeneratorConfig(seed=seed)
    sessions, utterances, entities, mentions, relationships, devices, labels, stats_dict = generate_structured_heterograph(config)
    # Builder expects session dict with id and started_at (optional community_id is fine)
    data = build_hetero_from_tables("hh1", sessions, utterances, entities, mentions, relationships, devices)
    try:
        node_types, edge_types = data.metadata()
    except Exception:
        node_types = [getattr(s, "key", i) for i, s in enumerate(data.node_stores)]
        edge_types = [getattr(s, "key", None) for s in data.edge_stores]
        edge_types = [e for e in edge_types if e is not None]
    in_channels = {nt: data[nt].x.size(1) for nt in node_types}
    entity_ids = [e["id"] for e in entities]
    n = len(entity_ids)
    data["entity"].y = torch.from_numpy(labels)
    # Train/val/test masks (same ratio as _add_synthetic_masks_and_labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(n * 0.7))
    n_val = max(0, int(n * 0.15))
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True
    data["entity"].train_mask = torch.from_numpy(train_mask)
    data["entity"].val_mask = torch.from_numpy(val_mask)
    data["entity"].test_mask = torch.from_numpy(test_mask)
    synthetic_config = {
        "source": "structured_synthetic",
        "counts": {"entities": len(entities), "sessions": len(sessions), "utterances": len(utterances), "mentions": len(mentions), "relationships": len(relationships)},
        "fraud_ratio": float(np.mean(labels == 1)),
        "seed": seed,
        **stats_dict,
    }
    graph_stats = _graph_stats(data, node_types, edge_types)
    graph_stats["structured_stats"] = stats_dict
    return in_channels, [data], synthetic_config, graph_stats, entity_ids


def _add_synthetic_masks_and_labels(
    data: HeteroData, target_node_type: str, seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> None:
    n = data[target_node_type].x.size(0)
    rng = np.random.default_rng(seed)
    y = np.array([0, 1, 0][:n], dtype=np.int64) if n <= 3 else rng.integers(0, 2, size=n)
    perm = rng.permutation(n)
    n_train, n_val = max(1, int(n * train_ratio)), max(0, int(n * val_ratio))
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train : n_train + n_val]] = True
    test_mask[perm[n_train + n_val :]] = True
    data[target_node_type].y = torch.from_numpy(y)
    data[target_node_type].train_mask = torch.from_numpy(train_mask)
    data[target_node_type].val_mask = torch.from_numpy(val_mask)
    data[target_node_type].test_mask = torch.from_numpy(test_mask)


def _graph_stats(data: HeteroData, node_types: list, edge_types: list) -> dict:
    stats: dict = {"node_counts": {}, "edge_counts": {}, "label_balance": None}
    for nt in node_types:
        try:
            stats["node_counts"][nt] = int(data[nt].x.size(0))
        except (KeyError, TypeError):
            pass
    for e in edge_types:
        try:
            key = e if isinstance(e, tuple) else (e[0], e[1], e[2])
            stats["edge_counts"][str(key)] = int(data[key].edge_index.size(1))
        except (KeyError, TypeError):
            pass
    if "entity" in node_types and getattr(data["entity"], "y", None) is not None:
        y = data["entity"].y.numpy()
        stats["label_balance"] = {"n": int(len(y)), "n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum())}
    return stats


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


@torch.no_grad()
def evaluate_hetero(
    model: nn.Module,
    data_list: list,
    device: torch.device,
    target_node_type: str | None = None,
    mask_names: tuple[str, ...] = ("train_mask", "val_mask", "test_mask"),
) -> dict[str, float]:
    """Run evaluation on hetero data. Uses per-node-type masks if present (e.g. HGB)."""
    model.eval()
    metrics: dict[str, float] = {}
    for data in data_list:
        data = data.to(device)
        out = model.forward_hetero_data(data)
        nt, labels = _get_hetero_labels(data, target_node_type)
        if nt is None or nt not in out or labels is None:
            continue
        labels = labels.to(device).long()
        node_out = out[nt]
        if node_out.size(0) != labels.size(0):
            continue
        pred = node_out.argmax(dim=-1)
        store = data[nt]
        for mask_name in mask_names:
            mask = getattr(store, mask_name, None)
            if mask is None or not getattr(mask, "any", lambda: False)():
                continue
            mask = mask.to(device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            y_m = labels[mask]
            pred_m = pred[mask]
            valid = y_m >= 0
            if not valid.any():
                continue
            y_m = y_m[valid]
            pred_m = pred_m[valid]
            acc = (pred_m == y_m).float().mean().item()
            key_acc = mask_name.replace("_mask", "_accuracy")
            metrics[key_acc] = acc
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(y_m.cpu().numpy(), pred_m.cpu().numpy(), average="macro", zero_division=0)
                metrics[mask_name.replace("_mask", "_macro_f1")] = float(f1)
            except Exception:
                pass
        # If no masks, report full-graph accuracy (valid labels only)
        if not metrics:
            valid = labels >= 0
            if valid.any():
                acc = (pred[valid] == labels[valid]).float().mean().item()
                metrics["accuracy"] = acc
                try:
                    from sklearn.metrics import f1_score
                    f1 = f1_score(labels[valid].cpu().numpy(), pred[valid].cpu().numpy(), average="macro", zero_division=0)
                    metrics["macro_f1"] = float(f1)
                except Exception:
                    pass
        break  # single graph for eval
    return metrics


@torch.no_grad()
def evaluate_hetero_with_preds(
    model: nn.Module,
    data_list: list,
    device: torch.device,
    target_node_type: str | None = None,
    mask_names: tuple[str, ...] = ("train_mask", "val_mask", "test_mask"),
) -> tuple[dict[str, float], dict[str, dict]]:
    """Run evaluation and return metrics plus per-mask y_true, logits, probs for PR/ROC curves."""
    model.eval()
    metrics: dict[str, float] = {}
    preds_by_mask: dict[str, dict] = {}
    for data in data_list:
        data = data.to(device)
        out = model.forward_hetero_data(data)
        nt, labels = _get_hetero_labels(data, target_node_type)
        if nt is None or nt not in out or labels is None:
            continue
        labels = labels.to(device).long()
        node_out = out[nt]
        if node_out.size(0) != labels.size(0):
            continue
        logits = node_out.cpu().numpy()
        probs = F.softmax(node_out, dim=-1).cpu().numpy()[:, 1]
        pred = node_out.argmax(dim=-1)
        y_np = labels.cpu().numpy()
        store = data[nt]
        for mask_name in mask_names:
            mask = getattr(store, mask_name, None)
            if mask is None or not getattr(mask, "any", lambda: False)():
                continue
            mask = mask.to(device).bool()
            y_m = labels[mask].cpu().numpy()
            valid = y_m >= 0
            if not valid.any():
                continue
            key = mask_name.replace("_mask", "")
            mask_np = mask.cpu().numpy()
            preds_by_mask[key] = {
                "y_true": y_m,
                "logits": logits[mask_np],
                "probs": probs[mask_np],
                "indices": np.where(mask_np)[0],
            }
            pred_m = pred[mask][valid].cpu().numpy()
            y_m = y_m[valid]
            acc = (pred_m == y_m).mean()
            metrics[f"{key}_accuracy"] = float(acc)
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score
                prob_pos = probs[mask.cpu().numpy()][valid]
                if np.unique(y_m).size >= 2:
                    metrics[f"{key}_pr_auc"] = float(average_precision_score(y_m, prob_pos))
                    metrics[f"{key}_roc_auc"] = float(roc_auc_score(y_m, prob_pos))
            except Exception:
                pass
        break
    return metrics, preds_by_mask


def _compute_motif_stats_hetero(data: HeteroData, target_nt: str, labels: np.ndarray) -> dict:
    """Compute motif stats (triangle count, degree distribution, etc.) for entity co_occurs, split by fraud/non-fraud."""
    out: dict = {"fraud": {}, "non_fraud": {}, "all": {}}
    try:
        key = ("entity", "co_occurs", "entity")
        ei = data[key].edge_index.numpy()
    except (KeyError, TypeError, AttributeError):
        return out
    n = data[target_nt].x.size(0)
    degree = np.zeros(n)
    for i in range(ei.shape[1]):
        degree[ei[0, i]] += 1
    out["all"]["degree_mean"] = float(np.mean(degree))
    out["all"]["degree_max"] = int(np.max(degree))
    out["all"]["num_edges"] = int(ei.shape[1])
    fraud_idx = labels == 1
    non_fraud_idx = labels == 0
    if fraud_idx.any():
        out["fraud"]["degree_mean"] = float(np.mean(degree[fraud_idx]))
        out["fraud"]["count"] = int(fraud_idx.sum())
    if non_fraud_idx.any():
        out["non_fraud"]["degree_mean"] = float(np.mean(degree[non_fraud_idx]))
        out["non_fraud"]["count"] = int(non_fraud_idx.sum())
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(ei.T.tolist())
        triangles = sum(nx.triangles(G).values()) // 3
        out["all"]["triangle_count"] = int(triangles)
    except Exception:
        out["all"]["triangle_count"] = 0
    return out


def _contrastive_loss_hetero(
    embed: torch.Tensor, labels: torch.Tensor, tau: float = 0.1
) -> torch.Tensor:
    """In-batch contrastive: pull same-label together, push different-label apart. embed: (N, D), labels: (N,)."""
    if embed.size(0) < 2:
        return embed.new_zeros(())
    z = F.normalize(embed, p=2, dim=-1)
    sim = z @ z.T  # (N, N)
    valid = labels >= 0
    if valid.sum() < 2:
        return embed.new_zeros(())
    mask_i = valid.unsqueeze(1)
    mask_j = valid.unsqueeze(0)
    same = (labels.unsqueeze(1) == labels.unsqueeze(0)) & mask_i & mask_j
    diff = (labels.unsqueeze(1) != labels.unsqueeze(0)) & mask_i & mask_j
    same = same.float()
    diff = diff.float()
    same = same * (1 - torch.eye(embed.size(0), device=embed.device))
    if same.sum() < 1 or diff.sum() < 1:
        return embed.new_zeros(())
    pos = (sim * same).sum() / (same.sum() + 1e-8)
    neg = (sim * diff).sum() / (diff.sum() + 1e-8)
    return -pos + neg


def train_step(
    model: nn.Module,
    data_list: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_hetero: bool = True,
    target_node_type: str | None = None,
    contrastive_weight: float = 0.0,
) -> tuple[float, float]:
    """Returns (mean_total_loss, mean_ce_loss). Log train_loss = mean_ce_loss for valid CE curves."""
    model.train()
    total_loss = 0.0
    ce_loss_sum = 0.0
    n_steps = 0
    for data in data_list:
        if use_hetero:
            data = data.to(device)
            use_contrastive = (
                contrastive_weight > 0
                and hasattr(model, "forward_hetero_data_with_hidden")
                and getattr(model, "embed_proj", None) is not None
            )
            if use_contrastive:
                out, h_dict, embed_dict = model.forward_hetero_data_with_hidden(data)
            else:
                out = model.forward_hetero_data(data)
                embed_dict = None
            nt, labels = _get_hetero_labels(data, target_node_type)
            if nt is None or nt not in out:
                continue
            node_out = out[nt]
            if labels is None:
                labels = torch.zeros(node_out.size(0), dtype=torch.long, device=device)
                if labels.size(0) > 0:
                    labels[0] = 0
                    if labels.size(0) > 1:
                        labels[1] = 1
            else:
                labels = labels.to(device).long()
            if node_out.size(0) != labels.size(0):
                continue
            train_mask = getattr(data[nt], "train_mask", None)
            if train_mask is not None and getattr(train_mask, "any", lambda: False)():
                train_mask = train_mask.to(device)
                if train_mask.dtype != torch.bool:
                    train_mask = train_mask.bool()
                valid = labels[train_mask] >= 0
                if valid.any():
                    loss = F.cross_entropy(node_out[train_mask][valid], labels[train_mask][valid])
                else:
                    loss = F.cross_entropy(node_out, labels)
            else:
                loss = F.cross_entropy(node_out, labels)
            ce_val = loss.item()
            assert ce_val >= -1e-6, f"CE loss must be non-negative, got {ce_val}"
            ce_loss_sum += ce_val
            n_steps += 1
            if use_contrastive and embed_dict is not None and nt in embed_dict:
                emb = embed_dict[nt]
                if emb.size(0) == labels.size(0):
                    cl = _contrastive_loss_hetero(emb, labels)
                    loss = loss + contrastive_weight * cl
        else:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            labels = getattr(data, "y", torch.zeros(out.size(0), dtype=torch.long, device=device))
            if out.size(0) != labels.size(0):
                labels = labels[: out.size(0)]
            loss = F.cross_entropy(out, labels)
            ce_loss_sum += loss.item()
            n_steps += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    n = max(len(data_list), 1)
    mean_total = total_loss / n
    mean_ce = ce_loss_sum / max(n_steps, 1)
    return mean_total, mean_ce


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/hgt_baseline.yaml")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "structured_synthetic", "hgb", "supabase_export"],
                        help="synthetic: minimal demo; structured_synthetic: fraud-from-structure; hgb: HGB; supabase_export: export script output")
    parser.add_argument("--hgb-name", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"],
                        help="HGB dataset name when --dataset hgb")
    parser.add_argument("--epochs", type=int, default=None, help="Override config train.epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override config train.lr")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None, help="Deprecated: use --run-dir")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Output run directory (default: runs/<timestamp>_<gitsha>)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    run_dir = Path(args.run_dir) if args.run_dir is not None else get_default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_train_config(args.config)
    train_cfg = cfg.get("train") or {}
    model_cfg = cfg.get("model") or {}
    graph_cfg = cfg.get("graph") or {}
    setup_run_dir(run_dir, cfg, list(sys.argv))

    entity_ids_list: list[str] | None = None
    synthetic_config: dict | None = None
    graph_stats: dict | None = None

    if args.dataset == "hgb":
        in_channels, data_list, metadata = get_hgb_hetero(args.data_dir, name=args.hgb_name)
        if not in_channels or not data_list:
            raise SystemExit("Failed to load HGB dataset. Check data-dir and network.")
        node_types_ok = [nt for nt in metadata[0] if nt in in_channels]
        edge_types_ok = [e for e in metadata[1] if e[0] in node_types_ok and e[2] in node_types_ok]
        metadata = (node_types_ok, edge_types_ok)
        target_node_type = None
        nt, labels = _get_hetero_labels(data_list[0], target_node_type)
        num_classes = 2
        if labels is not None and labels.numel() > 0:
            valid = labels >= 0
            if valid.any():
                num_classes = max(2, int(labels[valid].max().item()) + 1)
        logger.info("Loaded HGB %s: %d graphs, node types %s, num_classes %d", args.hgb_name, len(data_list), node_types_ok, num_classes)
    elif args.dataset == "structured_synthetic":
        in_channels, data_list, synthetic_config, graph_stats, entity_ids_list = get_structured_synthetic_hetero(args.data_dir, args.seed)
        try:
            node_types, edge_types = data_list[0].metadata()
        except Exception:
            node_types = list(in_channels.keys())
            edge_types = []
        metadata = (node_types, edge_types)
        target_node_type = "entity"
        num_classes = None
        with open(run_dir / "data_structured_synthetic.json", "w") as f:
            json.dump(_json_serialize(synthetic_config), f, indent=2)
        with open(run_dir / "graph_stats.json", "w") as f:
            json.dump(_json_serialize(graph_stats), f, indent=2)
        logger.info("Loaded structured synthetic: %d entities, fraud_ratio=%.3f, %s", len(entity_ids_list or []), synthetic_config.get("fraud_ratio", 0), graph_stats.get("structured_stats", {}))
    elif args.dataset == "supabase_export":
        in_channels, data_list, synthetic_config, graph_stats, entity_ids_list = get_supabase_export_hetero(args.data_dir, args.seed)
        try:
            node_types, edge_types = data_list[0].metadata()
        except Exception:
            node_types = list(in_channels.keys())
            edge_types = []
        metadata = (node_types, edge_types)
        target_node_type = "entity"
        num_classes = None
        with open(run_dir / "data_export.json", "w") as f:
            json.dump(synthetic_config, f, indent=2)
        with open(run_dir / "graph_stats.json", "w") as f:
            json.dump(graph_stats, f, indent=2)
        logger.info("Loaded Supabase export: %d entities, %s", graph_stats.get("num_entities"), graph_stats.get("num_entity_edges"))
    else:
        in_channels, data_list, synthetic_config, graph_stats, entity_ids_list = get_synthetic_hetero(args.data_dir, args.seed)
        node_types = graph_cfg.get("node_types", ["person", "device", "session", "event", "utterance", "intent", "entity"])
        edge_types_raw = graph_cfg.get("edge_types", [])
        edge_types = [tuple(e) if isinstance(e, list) else e for e in edge_types_raw]
        metadata = (node_types, edge_types)
        target_node_type = "entity"
        num_classes = None
        with open(run_dir / "data_synthetic.json", "w") as f:
            json.dump(synthetic_config, f, indent=2)
        with open(run_dir / "graph_stats.json", "w") as f:
            json.dump(graph_stats, f, indent=2)

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 50)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)
    hidden = model_cfg.get("hidden_channels", 32)
    out_channels = int(num_classes) if num_classes is not None else model_cfg.get("out_channels", 2)
    num_layers = model_cfg.get("num_layers", 2)
    heads = model_cfg.get("heads", 4)
    embed_dim = model_cfg.get("embed_dim", 128)
    contrastive_weight = float(train_cfg.get("contrastive_weight", 0.0))

    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=out_channels,
        num_layers=num_layers,
        heads=heads,
        metadata=metadata,
        embed_dim=embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history_file = run_dir / "history.jsonl"
    row: dict = {}
    warned_negative = False
    # Fresh file each run: exactly one row per epoch (no duplicate epochs from prior runs)
    history_file.write_text("")
    for epoch in range(epochs):
        total_loss, ce_loss = train_step(
            model, data_list, optimizer, device,
            use_hetero=True, target_node_type=target_node_type,
            contrastive_weight=contrastive_weight,
        )
        if total_loss < 0 and not warned_negative:
            logger.warning(
                "Total train loss went negative (%.4f) due to contrastive term; logging CE-only as train_loss for valid curve.",
                total_loss,
            )
            warned_negative = True
        row = {"epoch": epoch + 1, "train_loss": round(ce_loss, 6)}
        if total_loss != ce_loss:
            row["train_total_loss"] = round(total_loss, 6)
        eval_metrics, preds_by_mask = evaluate_hetero_with_preds(model, data_list, device, target_node_type=target_node_type)
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float)):
                row[k] = round(v, 6) if isinstance(v, float) else v
        with open(history_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d loss %.4f val_pr_auc %s val_roc_auc %s", epoch + 1, ce_loss, row.get("val_pr_auc"), row.get("val_roc_auc"))

    eval_metrics, preds_by_mask = evaluate_hetero_with_preds(model, data_list, device, target_node_type=target_node_type)
    if eval_metrics:
        logger.info("Eval: %s", json.dumps(eval_metrics, indent=2))

    label_counts_run: dict = {}
    for key, preds in preds_by_mask.items():
        arrs = {"y_true": preds["y_true"], "logits": preds["logits"], "probs": preds["probs"]}
        if entity_ids_list is not None and "indices" in preds:
            arrs["entity_ids"] = np.array([entity_ids_list[i] for i in preds["indices"]])
        np.savez(run_dir / f"preds_{key}.npz", **arrs)
        y = preds["y_true"]
        valid = y >= 0
        if valid.any():
            y_v = y[valid]
            pos, neg = int((y_v == 1).sum()), int((y_v == 0).sum())
            label_counts_run[key] = {"pos": pos, "neg": neg, "n": int(valid.sum())}
    if label_counts_run:
        with open(run_dir / "label_counts.json", "w") as f:
            json.dump(label_counts_run, f, indent=2)

    calibration: dict = {}
    if "val" in preds_by_mask and preds_by_mask["val"]["y_true"].size > 0:
        try:
            from sklearn.linear_model import LogisticRegression
            y_val = preds_by_mask["val"]["y_true"]
            logits_val = preds_by_mask["val"]["logits"]
            if logits_val.ndim == 2:
                logit_pos = logits_val[:, 1]
            else:
                logit_pos = logits_val
            lr_cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=500)
            lr_cal.fit(logit_pos.reshape(-1, 1), y_val)
            a, b = float(lr_cal.coef_[0, 0]), float(lr_cal.intercept_[0])
            calibration["platt_a"] = a
            calibration["platt_b"] = b
            calibration["calibration_set_size"] = int(y_val.size)
            probs_cal = 1.0 / (1.0 + np.exp(-(a * logit_pos + b)))
            from sklearn.metrics import brier_score_loss
            calibration["brier_pre"] = float(brier_score_loss(y_val, preds_by_mask["val"]["probs"]))
            calibration["brier_post"] = float(brier_score_loss(y_val, probs_cal))
        except Exception as e:
            calibration["error"] = str(e)
        with open(run_dir / "calibration.json", "w") as f:
            json.dump(calibration, f, indent=2)

    data = data_list[0]
    data = data.to(device)
    if hasattr(model, "forward_hetero_data_with_hidden"):
        with torch.no_grad():
            _, _, embed_dict = model.forward_hetero_data_with_hidden(data)
            if embed_dict and target_node_type in embed_dict:
                emb = embed_dict[target_node_type].cpu().numpy()
                labels_ent = data[target_node_type].y.cpu().numpy()
                out_ent = model.forward_hetero_data(data)[target_node_type]
                probs_ent = F.softmax(out_ent, dim=-1)[:, 1].detach().cpu().numpy()
        if embed_dict and target_node_type in embed_dict:
            np.savez(run_dir / "embeddings_entity.npz", embeddings=emb, labels=labels_ent, probs=probs_ent)
            with open(run_dir / "embeddings_entity_meta.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["entity_id", "label", "prob", "degree"])
                key_co = ("entity", "co_occurs", "entity")
                degree = np.zeros(emb.shape[0])
                try:
                    ei = data[key_co].edge_index.cpu().numpy()
                    for j in range(ei.shape[1]):
                        degree[ei[0, j]] += 1
                except (KeyError, TypeError):
                    pass
                for i in range(emb.shape[0]):
                    eid = entity_ids_list[i] if entity_ids_list and i < len(entity_ids_list) else str(i)
                    w.writerow([eid, int(labels_ent[i]), round(float(probs_ent[i]), 6), int(degree[i])])

    ex_dir = run_dir / "explanations"
    ex_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        out = model.forward_hetero_data(data)
    nt = target_node_type
    probs = F.softmax(out[nt], dim=-1)[:, 1].cpu().numpy()
    top_k = min(25, probs.size)
    top_indices = np.argsort(-probs)[:top_k]
    key_co = ("entity", "co_occurs", "entity")
    try:
        edge_index_co = data[key_co].edge_index.cpu().numpy()
    except (KeyError, TypeError):
        edge_index_co = np.zeros((2, 0))
    top_entities_list = []
    for idx in top_indices:
        node_set = {int(idx)}
        for j in range(edge_index_co.shape[1]):
            a, b = int(edge_index_co[0, j]), int(edge_index_co[1, j])
            if a == idx or b == idx:
                node_set.add(a)
                node_set.add(b)
        nodes = [{"id": entity_ids_list[i] if entity_ids_list and i < len(entity_ids_list) else str(i), "type": "entity"} for i in node_set]
        edges = []
        for j in range(edge_index_co.shape[1]):
            a, b = int(edge_index_co[0, j]), int(edge_index_co[1, j])
            if a in node_set and b in node_set:
                edges.append({
                    "src": entity_ids_list[a] if entity_ids_list and a < len(entity_ids_list) else str(a),
                    "dst": entity_ids_list[b] if entity_ids_list and b < len(entity_ids_list) else str(b),
                    "src_type": "entity", "rel": "co_occurs", "dst_type": "entity",
                    "weight": 1.0, "hop": 1,
                })
        eid = entity_ids_list[int(idx)] if entity_ids_list and int(idx) < len(entity_ids_list) else str(int(idx))
        rec = {"entity_id": eid, "prob": round(float(probs[idx]), 6), "nodes": nodes, "edges": edges}
        top_entities_list.append(rec)
        with open(ex_dir / f"subgraph_{eid}.json", "w") as f:
            json.dump(rec, f, indent=2)
    with open(ex_dir / "top_entities.json", "w") as f:
        json.dump(top_entities_list, f, indent=2)

    if target_node_type and getattr(data[target_node_type], "y", None) is not None:
        labels_np = data[target_node_type].y.cpu().numpy()
        num_entities = data[target_node_type].x.size(0)
        try:
            num_entity_edges = int(data[("entity", "co_occurs", "entity")].edge_index.size(1))
        except (KeyError, TypeError):
            num_entity_edges = 0
        if num_entities < 50 or num_entity_edges < 200:
            logger.warning(
                "Whitepaper minimum not met: num_entities=%d (>=50), num_entity_edges=%d (>=200). "
                "Use a larger synthetic graph or multi-session data for meaningful ROC/PR/motifs.",
                num_entities, num_entity_edges,
            )
        gs = dict(graph_stats) if graph_stats else {}
        gs["num_entities"] = num_entities
        gs["num_entity_edges"] = num_entity_edges
        with open(run_dir / "graph_stats.json", "w") as f:
            json.dump(gs, f, indent=2)
        motifs = _compute_motif_stats_hetero(data, target_node_type, labels_np)
        with open(run_dir / "motifs.json", "w") as f:
            json.dump(motifs, f, indent=2)

    final_metrics = {**eval_metrics, "train_final_loss": row.get("train_loss")}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    ckpt = {
        "model_state": model.state_dict(),
        "metadata": metadata,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "hidden_channels": hidden,
        "num_layers": num_layers,
        "heads": heads,
        "embed_dim": embed_dim,
        "target_node_type": target_node_type,
    }
    torch.save(ckpt, run_dir / "best.pt")
    logger.info("Saved to %s", run_dir / "best.pt")


if __name__ == "__main__":
    main()
