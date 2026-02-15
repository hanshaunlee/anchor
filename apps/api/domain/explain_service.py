"""Explanation domain: subgraph from explanation JSON, similar incidents, deep-dive explainer."""
from __future__ import annotations

from uuid import UUID

from supabase import Client

from api.schemas import (
    RiskSignalDetailSubgraph,
    SimilarIncidentsResponse,
    SubgraphEdge,
    SubgraphNode,
)
from domain.similarity_service import get_similar_incidents as get_similar_incidents_impl


def build_subgraph_from_explanation(
    explanation: dict,
    *,
    prefer_key: str | None = None,
) -> RiskSignalDetailSubgraph | None:
    """Build RiskSignalDetailSubgraph from explanation dict.
    Uses prefer_key first if present (e.g. 'deep_dive_subgraph'), else subgraph or model_subgraph."""
    subgraph_data = {}
    if prefer_key and explanation.get(prefer_key):
        subgraph_data = explanation.get(prefer_key) or {}
    if not subgraph_data:
        subgraph_data = explanation.get("subgraph") or explanation.get("model_subgraph") or {}
    nodes = [
        SubgraphNode(
            id=str(n.get("id", "")),
            type=n.get("type", ""),
            label=n.get("label"),
            score=n.get("score"),
        )
        for n in subgraph_data.get("nodes", [])
    ]
    edges = [
        SubgraphEdge(
            src=e["src"],
            dst=e["dst"],
            type=e.get("type", ""),
            weight=e.get("weight") or e.get("importance"),
            rank=e.get("rank"),
        )
        for e in subgraph_data.get("edges", [])
    ]
    return RiskSignalDetailSubgraph(nodes=nodes, edges=edges) if (nodes or edges) else None


def get_similar_incidents(
    signal_id: UUID,
    household_id: str,
    supabase: Client,
    top_k: int = 5,
) -> SimilarIncidentsResponse:
    """Retrieve nearest neighbors by real embedding (cosine). Delegates to similarity_service."""
    return get_similar_incidents_impl(signal_id, household_id, supabase, top_k=top_k)


def run_deep_dive_explainer(
    signal_id: UUID,
    household_id: str,
    supabase: Client,
    mode: str = "pg",
) -> dict:
    """
    Run deep-dive explainer (PG or GNN) and persist deep_dive_subgraph on the risk_signal explanation.
    mode=pg: copies model_subgraph to deep_dive_subgraph (PGExplainer result). mode=gnn: not yet implemented.
    Returns { "ok": True, "method": "pg", "deep_dive_subgraph": {...} } or raises for 404/501.
    """
    r = (
        supabase.table("risk_signals")
        .select("id, explanation")
        .eq("id", str(signal_id))
        .eq("household_id", household_id)
        .single()
        .execute()
    )
    if not r.data:
        raise ValueError("Risk signal not found")
    expl = dict(r.data.get("explanation") or {})
    if mode == "gnn":
        # GNNExplainer requires hetero-model adapter; not yet wired.
        raise NotImplementedError("Deep dive mode=gnn not yet implemented (hetero model adapter required)")
    if mode == "pg":
        model_sg = expl.get("model_subgraph")
        if not model_sg or not (model_sg.get("nodes") or model_sg.get("edges")):
            raise ValueError("No model subgraph available for this signal (model may not have run)")
        deep_dive = {
            "nodes": list(model_sg.get("nodes", [])),
            "edges": list(model_sg.get("edges", [])),
            "method": "pg",
        }
        expl["deep_dive_subgraph"] = deep_dive
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        supabase.table("risk_signals").update({"explanation": expl, "updated_at": now_iso}).eq("id", str(signal_id)).eq("household_id", household_id).execute()
        return {"ok": True, "method": "pg", "deep_dive_subgraph": deep_dive}
    raise ValueError("mode must be 'pg' or 'gnn'")
