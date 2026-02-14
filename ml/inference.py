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


def load_model(checkpoint_path: Path, device: torch.device) -> HGTBaseline:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = ckpt["in_channels"]
    metadata = ckpt.get("metadata")
    model = HGTBaseline(
        in_channels=in_channels,
        hidden_channels=32,
        out_channels=2,
        num_layers=2,
        heads=4,
        metadata=metadata,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: HGTBaseline,
    data,
    device: torch.device,
    explain_node_idx: int | None = None,
) -> tuple[list[dict], dict | None]:
    """Return list of risk items (node_id, score, label) and optional explanation_json."""
    data = data.to(device)
    with torch.no_grad():
        out = model.forward_hetero_data(data)
    entity_out = out["entity"]
    probs = torch.softmax(entity_out, dim=-1)
    risk_list = []
    for i in range(entity_out.size(0)):
        risk_list.append({
            "node_type": "entity",
            "node_index": i,
            "score": float(probs[i, 1]),
            "label": int(probs[i, 1] > 0.5),
        })
    explanation_json = None
    if explain_node_idx is not None and entity_out.size(0) > explain_node_idx:
        # Build homogeneous entity subgraph for explainer
        from torch_geometric.data import Data
        if hasattr(data["entity", "co_occurs", "entity"], "edge_index"):
            edge_index = data["entity", "co_occurs", "entity"].edge_index
            edge_attr = data["entity", "co_occurs", "entity"].edge_attr
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 4)
        x = data["entity"].x
        hom_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # Simple wrapper model for explainer (entity logits only)
        class Wrapper(torch.nn.Module):
            def __init__(self, hgt_model):
                super().__init__()
                self.hgt = hgt_model
            def forward(self, x, edge_index, edge_attr=None):
                # Minimal forward: we only need entity logits; use full hetero for real scoring
                return self.hgt.forward_hetero_data(data)["entity"]
        wrapper = Wrapper(model)
        explanation_json = explain_node_gnn_explainer_style(
            wrapper,
            hom_data,
            target_node=explain_node_idx,
            num_edges=edge_index.size(1),
            epochs=50,
        )
    return risk_list, explanation_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/hgt_baseline/best.pt"))
    parser.add_argument("--household-id", type=str, default="hh1")
    parser.add_argument("--explain-node", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cpu")

    model = load_model(args.checkpoint, device)
    # Build minimal graph like in train
    from ml.train import get_synthetic_hetero
    in_channels, data_list = get_synthetic_hetero(Path("data/synthetic"))
    data = data_list[0]
    risk_list, explanation_json = run_inference(model, data, device, explain_node_idx=args.explain_node)
    print(json.dumps({"risk_scores": risk_list, "explanation": explanation_json}, indent=2))


if __name__ == "__main__":
    main()
