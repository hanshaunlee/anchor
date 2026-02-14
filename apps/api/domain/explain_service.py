"""Explanation domain: subgraph from explanation JSON, similar incidents by embedding."""
from uuid import UUID

from supabase import Client

from api.schemas import (
    RiskSignalDetailSubgraph,
    SimilarIncidentsResponse,
    SubgraphEdge,
    SubgraphNode,
)
from domain.similarity_service import get_similar_incidents as get_similar_incidents_impl


def build_subgraph_from_explanation(explanation: dict) -> RiskSignalDetailSubgraph | None:
    """Build RiskSignalDetailSubgraph from explanation dict (subgraph or model_subgraph)."""
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
            weight=e.get("weight"),
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
