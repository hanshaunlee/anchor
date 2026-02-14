"""
Single risk scoring service: one function, one response schema.
Used by pipeline, worker, and Financial Security Agent. No silent placeholder scores;
when model is unavailable, returns model_available=False and empty scores (callers may add explicit rule-only fallback).
"""
from __future__ import annotations

import logging
from typing import Any

from api.schemas import RiskScoreItem, RiskScoringResponse

logger = logging.getLogger(__name__)


def _default_checkpoint_path():
    from pathlib import Path
    try:
        from config.settings import get_ml_settings
        return Path(get_ml_settings().checkpoint_path)
    except Exception:
        return Path("runs/hgt_baseline/best.pt")


def _edge_attr_dim() -> int:
    try:
        from config.graph import get_graph_config
        return get_graph_config().get("edge_attr_dim", 4)
    except ImportError:
        return 4


def _pipeline_settings():
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings()
    except ImportError:
        class _F:
            explanation_score_min = 0.4
        return _F()


def score_risk(
    household_id: str,
    *,
    sessions: list[dict],
    utterances: list[dict],
    entities: list[dict],
    mentions: list[dict],
    relationships: list[dict],
    devices: list[dict] | None = None,
    events: list[dict] | None = None,
    checkpoint_path: Any = None,
    explanation_score_min: float | None = None,
) -> RiskScoringResponse:
    """
    Run GNN risk scoring on the given graph context. Single contract for pipeline, worker, and agent.
    Returns model_available=True and scores (with embeddings and model_subgraph when above threshold)
    when the checkpoint exists and inference succeeds; otherwise model_available=False and scores=[].
    Callers must not fabricate placeholder scores; they may add explicit rule-only fallback with fallback_used set.
    """
    if not entities:
        return RiskScoringResponse(model_available=False, scores=[])

    devices = devices or []
    events = events or []
    path = checkpoint_path
    if path is None:
        path = _default_checkpoint_path()
    if hasattr(path, "is_file") and not path.is_file():
        logger.debug("Checkpoint missing at %s", path)
        return RiskScoringResponse(model_available=False, scores=[])

    try:
        import torch
        from ml.inference import load_model, run_inference
        from ml.graph.builder import build_hetero_from_tables
    except ImportError as e:
        logger.debug("ML stack unavailable: %s", e)
        return RiskScoringResponse(model_available=False, scores=[])

    try:
        device = torch.device("cpu")
        model, target_node_type = load_model(path, device)
        data = build_hetero_from_tables(
            household_id,
            sessions,
            utterances,
            entities,
            mentions,
            relationships,
            devices=devices,
        )
        raw_scores, _ = run_inference(
            model, data, device,
            target_node_type=target_node_type,
            return_embeddings=True,
        )
    except Exception as e:
        logger.debug("Inference failed: %s", e)
        return RiskScoringResponse(model_available=False, scores=[])

    # Attach PGExplainer model_subgraph for nodes above threshold (inline to avoid circular import with pipeline)
    expl_min = explanation_score_min if explanation_score_min is not None else _pipeline_settings().explanation_score_min
    try:
        import torch as _torch
        from torch_geometric.data import Data as _Data
        from ml.explainers.pg_explainer import PGExplainerStyle, explain_with_pg
    except ImportError as e:
        logger.debug("PGExplainer not available: %s", e)
    else:
        with _torch.no_grad():
            _, h_dict = model.forward_hetero_data_with_hidden(data)
        node_emb = h_dict.get(target_node_type or "entity")
        if node_emb is not None and node_emb.size(0) > 0:
            edge_index = _torch.empty(2, 0, dtype=_torch.long, device=device)
            edge_attr = None
            try:
                _, edge_types = data.metadata()
                for (src, rel, dst) in edge_types:
                    if src == (target_node_type or "entity") and dst == (target_node_type or "entity"):
                        store = data[src, rel, dst]
                        edge_index = store.edge_index.to(device)
                        edge_attr = getattr(store, "edge_attr", None)
                        if edge_attr is not None:
                            edge_attr = edge_attr.to(device)
                        break
            except Exception:
                pass
            edge_dim = _edge_attr_dim()
            if edge_attr is None and edge_index.size(1) > 0:
                edge_attr = _torch.zeros(edge_index.size(1), edge_dim, device=device)
            elif edge_attr is not None:
                edge_dim = edge_attr.size(-1)
            hom_data = _Data(
                x=data[target_node_type or "entity"].x.to(device),
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            hidden_dim = node_emb.size(-1)
            pg = PGExplainerStyle(hidden_dim, edge_dim).to(device).eval()
            score_by_idx = {r["node_index"]: r["score"] for r in raw_scores}
            total_edges = edge_index.size(1)
            top_k_edges = 20

            def _entity_id(n: int) -> str:
                if n < len(entities) and entities[n].get("id") is not None:
                    return str(entities[n]["id"])
                return str(n)

            for r in raw_scores:
                if r.get("score", 0) < expl_min:
                    continue
                node_idx = r.get("node_index", 0)
                if edge_index.size(1) == 0:
                    r["model_subgraph"] = {
                        "nodes": [{"id": _entity_id(node_idx), "type": "entity", "score": r.get("score", 0)}],
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
                    {"id": _entity_id(n), "type": "entity", "score": score_by_idx.get(n) if n < len(raw_scores) else None}
                    for n in sorted(incident_nodes)
                ]
                edges = [
                    {"src": _entity_id(e["src"]), "dst": _entity_id(e["dst"]), "weight": round(e["score"], 4), "rank": i}
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

    # Build response with explicit model_available=True (all came from GNN)
    scores_list: list[RiskScoreItem] = []
    for r in raw_scores:
        item = RiskScoreItem(
            node_type=r.get("node_type", "entity"),
            node_index=r["node_index"],
            score=r["score"],
            signal_type=r.get("signal_type", "relational_anomaly"),
            embedding=r.get("embedding") if isinstance(r.get("embedding"), (list, tuple)) else None,
            model_subgraph=r.get("model_subgraph"),
            model_available=True,
        )
        scores_list.append(item)

    return RiskScoringResponse(model_available=True, scores=scores_list)
