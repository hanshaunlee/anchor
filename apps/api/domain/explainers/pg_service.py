"""
Single source of truth for PGExplainer: build homogeneous graph, run explainer,
map PyG indices back to entity IDs, attach model_subgraph to risk_scores.
All model_subgraph nodes/edges use entity_id (never raw PyG index).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _edge_attr_dim() -> int:
    try:
        from config.graph import get_graph_config
        return get_graph_config().get("edge_attr_dim", 4)
    except ImportError:
        return 4


def _entity_id_from_index(node_index: int, entities: list[dict]) -> str:
    """Map PyG node index to entity id for explanation JSON."""
    if 0 <= node_index < len(entities) and entities[node_index].get("id") is not None:
        return str(entities[node_index]["id"])
    return str(node_index)


def attach_pg_explanations(
    model: Any,
    hetero_data: Any,
    risk_scores: list[dict],
    entities: list[dict],
    *,
    target_node_type: str = "entity",
    explanation_score_min: float = 0.4,
    top_k_edges: int = 20,
    device: Any = None,
) -> None:
    """
    Run PGExplainer for each risk score above threshold; attach model_subgraph
    with node/edge IDs as entity IDs (not PyG indices). Mutates risk_scores in place.
    """
    try:
        import torch
        from torch_geometric.data import Data
        from ml.explainers.pg_explainer import PGExplainerStyle, explain_with_pg
    except ImportError as e:
        logger.debug("PGExplainer not available: %s", e)
        return

    if device is None:
        device = torch.device("cpu")

    with torch.no_grad():
        _, h_dict = model.forward_hetero_data_with_hidden(hetero_data)
    node_emb = h_dict.get(target_node_type)
    if node_emb is None or node_emb.size(0) == 0:
        return

    edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    edge_attr = None
    try:
        _, edge_types = hetero_data.metadata()
        for (src, rel, dst) in edge_types:
            if src == target_node_type and dst == target_node_type:
                store = hetero_data[src, rel, dst]
                edge_index = store.edge_index.to(device)
                edge_attr = getattr(store, "edge_attr", None)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)
                break
    except Exception:
        pass
    edge_dim = _edge_attr_dim()
    if edge_attr is None and edge_index.size(1) > 0:
        edge_attr = torch.zeros(edge_index.size(1), edge_dim, device=device)
    elif edge_attr is not None:
        edge_dim = edge_attr.size(-1)
    hom_data = Data(
        x=hetero_data[target_node_type].x.to(device),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    hidden_dim = node_emb.size(-1)
    pg = PGExplainerStyle(hidden_dim, edge_dim).to(device).eval()
    score_by_idx = {r["node_index"]: r.get("score", 0) for r in risk_scores}
    total_edges = edge_index.size(1)

    for r in risk_scores:
        if r.get("score", 0) < explanation_score_min:
            continue
        node_idx = r.get("node_index", 0)
        if edge_index.size(1) == 0:
            r["model_subgraph"] = {
                "nodes": [{"id": _entity_id_from_index(node_idx, entities), "type": "entity", "score": r.get("score", 0)}],
                "edges": [],
            }
            r["model_evidence_quality"] = {"sparsity": 0.0, "edges_kept": 0, "edges_total": 0}
            r["model_available"] = True
            continue
        expl = explain_with_pg(pg, node_emb, hom_data, top_k=top_k_edges)
        top_edges = expl["top_edges"]
        incident_edges = [e for e in top_edges if e["src"] == node_idx or e["dst"] == node_idx]
        if not incident_edges:
            incident_edges = top_edges[:5]
        incident_nodes = set()
        for e in incident_edges:
            incident_nodes.add(e["src"])
            incident_nodes.add(e["dst"])
        if node_idx not in incident_nodes:
            incident_nodes.add(node_idx)
        nodes = [
            {"id": _entity_id_from_index(n, entities), "type": "entity", "score": score_by_idx.get(n)}
            for n in sorted(incident_nodes)
        ]
        edges = [
            {
                "src": _entity_id_from_index(e["src"], entities),
                "dst": _entity_id_from_index(e["dst"], entities),
                "weight": round(e["score"], 4),
                "importance": round(e["score"], 4),
                "rank": i,
            }
            for i, e in enumerate(incident_edges)
        ]
        r["model_subgraph"] = {"nodes": nodes, "edges": edges}
        edges_kept = len(incident_edges)
        r["model_evidence_quality"] = {
            "sparsity": round(1.0 - (edges_kept / total_edges) if total_edges else 0.0, 4),
            "edges_kept": edges_kept,
            "edges_total": total_edges,
        }
        r["model_available"] = True
