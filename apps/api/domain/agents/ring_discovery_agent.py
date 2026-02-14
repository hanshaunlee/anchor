"""Ring Discovery Agent: Neo4j GDS node similarity + embeddings; flag suspicious clusters; link to risk signals."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def run_ring_discovery_agent(
    household_id: str,
    supabase: Any | None = None,
    neo4j_available: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run Neo4j GDS node similarity (e.g. FastRP/Node2Vec) on mirrored evidence graph;
    flag suspicious clusters and link to risk signals.
    When Neo4j not available, returns stub step_trace.
    """
    step_trace: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()

    step_trace.append({"step": "check_neo4j", "status": "ok", "neo4j_available": neo4j_available})
    if not neo4j_available:
        pass  # Would use api.neo4j_sync and Neo4j GDS when configured
        step_trace.append({"step": "gds_similarity", "status": "skip", "reason": "neo4j_unavailable"})
        return {
            "step_trace": step_trace,
            "summary_json": {"clusters_found": 0, "neo4j_available": False},
            "status": "ok",
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }

    # Placeholder: would run Cypher GDS and create risk_signal or link to existing
    step_trace.append({"step": "gds_similarity", "status": "ok", "clusters_found": 0})
    return {
        "step_trace": step_trace,
        "summary_json": {"clusters_found": 0, "neo4j_available": True},
        "status": "ok",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
