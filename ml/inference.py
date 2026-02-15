#!/usr/bin/env python3
"""
Run risk scoring and optionally explanation for a household graph.
Loads checkpoint, builds graph from DB or in-memory, outputs risk_signal payloads.
"""
import argparse
import json
from pathlib import Path
from uuid import UUID

import torch

from ml.graph.builder import GraphBuilder, build_hetero_from_tables
from ml.models.hgt_baseline import HGTBaseline
from ml.explainers.gnn_explainer import explain_node_gnn_explainer_style


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HGTBaseline, str | None]:
    """Load HGT from checkpoint. Returns (model, target_node_type)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = ckpt["in_channels"]
    metadata = ckpt.get("metadata")
    hidden = ckpt.get("hidden_channels", 32)
    out_channels = ckpt.get("out_channels", 2)
    num_layers = ckpt.get("num_layers", 2)
    heads = ckpt.get("heads", 4)
    embed_dim = ckpt.get("embed_dim", 128)
    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=out_channels,
        num_layers=num_layers,
        heads=heads,
        metadata=metadata,
        embed_dim=embed_dim,
    )
    # strict=False so old checkpoints without embed_proj still load; embed_proj stays randomly init
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    target_node_type = ckpt.get("target_node_type")
    return model, target_node_type


def _edge_attr_dim() -> int:
    """Edge attribute dimension for empty fallbacks (from config)."""
    try:
        from config.graph import get_graph_config
        return get_graph_config().get("edge_attr_dim", 4)
    except ImportError:
        return 4


def _explainer_epochs() -> int:
    """Explainer optimization epochs (from config)."""
    try:
        from config.settings import get_ml_settings
        return get_ml_settings().explainer_epochs
    except ImportError:
        return 50


def run_inference(
    model: HGTBaseline,
    data,
    device: torch.device,
    target_node_type: str | None = None,
    explain_node_idx: int | None = None,
    explainer_epochs_override: int | None = None,
    return_embeddings: bool = True,
) -> tuple[list[dict], dict | None]:
    """Return list of risk items (node_id, score, label) and optional explanation_json.
    target_node_type: output node type (default 'entity'; use checkpoint value for HGB).
    return_embeddings: if True, attach model pooled representation (hidden before lin_out) as 'embedding' per node for retrieval."""
    data = data.to(device)
    out_node_type = target_node_type or "entity"
    with torch.no_grad():
        if return_embeddings and hasattr(model, "forward_hetero_data_with_hidden"):
            result = model.forward_hetero_data_with_hidden(data)
            out = result[0]
            h_dict = result[1]
            embed_dict = result[2] if len(result) > 2 else None
            # Prefer retrieval projection (embed_dict) when present; else classifier hidden
            if embed_dict is not None and out_node_type in embed_dict:
                hidden = embed_dict[out_node_type]
            else:
                hidden = h_dict.get(out_node_type) if h_dict else None
        else:
            out = model.forward_hetero_data(data)
            h_dict = None
            embed_dict = None
            hidden = None
    if out_node_type not in out:
        out_node_type = next(iter(out.keys()))
        if embed_dict is not None and out_node_type in embed_dict:
            hidden = embed_dict[out_node_type]
        elif h_dict is not None:
            hidden = h_dict.get(out_node_type)
    node_out = out[out_node_type]
    probs = torch.softmax(node_out, dim=-1)
    risk_list = []
    for i in range(node_out.size(0)):
        item = {
            "node_type": out_node_type,
            "node_index": i,
            "score": float(probs[i, 1]) if probs.size(1) > 1 else float(probs[i, 0]),
            "label": int(probs[i, 1] > 0.5) if probs.size(1) > 1 else 0,
        }
        if hidden is not None and i < hidden.size(0):
            item["embedding"] = hidden[i].cpu().float().tolist()
        risk_list.append(item)
    explanation_json = None
    edge_dim = _edge_attr_dim()
    if explain_node_idx is not None and node_out.size(0) > explain_node_idx:
        from torch_geometric.data import Data
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_attr = torch.empty(0, edge_dim)
        try:
            _, edge_types = data.metadata()
            for (src, rel, dst) in edge_types:
                if src == out_node_type and dst == out_node_type:
                    store = data[src, rel, dst]
                    edge_index = store.edge_index
                    edge_attr = getattr(store, "edge_attr", torch.empty(edge_index.size(1), edge_dim))
                    break
        except Exception:
            pass
        x = data[out_node_type].x
        hom_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        class Wrapper(torch.nn.Module):
            def __init__(self, hgt_model):
                super().__init__()
                self.hgt = hgt_model
            def forward(self, x, edge_index, edge_attr=None):
                return self.hgt.forward_hetero_data(data)[out_node_type]
        wrapper = Wrapper(model)
        epochs = explainer_epochs_override if explainer_epochs_override is not None else _explainer_epochs()
        explanation_json = explain_node_gnn_explainer_style(
            wrapper,
            hom_data,
            target_node=explain_node_idx,
            num_edges=edge_index.size(1),
            epochs=epochs,
        )
    return risk_list, explanation_json


def _default_checkpoint_path() -> Path:
    try:
        from config.settings import get_ml_settings
        return Path(get_ml_settings().checkpoint_path)
    except Exception:
        return Path("runs/hgt_baseline/best.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None, help="HGT checkpoint (default from ANCHOR_ML_CHECKPOINT_PATH)")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "hgb"],
                        help="Graph to run on: synthetic (entity graph) or hgb (must match checkpoint if you trained on HGB)")
    parser.add_argument("--hgb-name", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB", "Freebase"],
                        help="HGB dataset name when --dataset hgb")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"), help="Data dir for synthetic or HGB root")
    parser.add_argument("--household-id", type=str, default="hh1")
    parser.add_argument("--explain-node", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=0, help="Only output top K nodes by score (0 = all)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write JSON to file instead of stdout")
    args = parser.parse_args()
    device = torch.device("cpu")
    checkpoint = args.checkpoint if args.checkpoint is not None else _default_checkpoint_path()

    model, target_node_type = load_model(checkpoint, device)
    if args.dataset == "hgb":
        from ml.train import get_hgb_hetero
        _, data_list, _ = get_hgb_hetero(args.data_dir, name=args.hgb_name)
        if not data_list:
            raise SystemExit("Failed to load HGB dataset. Install torch-geometric and use --data-dir.")
        data = data_list[0]
    else:
        from ml.train import get_synthetic_hetero
        _, data_list = get_synthetic_hetero(args.data_dir)
        data = data_list[0]
    risk_list, explanation_json = run_inference(
        model, data, device, target_node_type=target_node_type, explain_node_idx=args.explain_node
    )
    if args.top_k > 0:
        risk_list = sorted(risk_list, key=lambda r: r["score"], reverse=True)[: args.top_k]
    payload = {"risk_scores": risk_list, "explanation": explanation_json}
    out = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out)
    else:
        print(out)


if __name__ == "__main__":
    main()
