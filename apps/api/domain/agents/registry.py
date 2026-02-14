"""
Agent registry: list all agents and metadata for /agents/status and UI.
"""
from __future__ import annotations

from typing import Any, Callable

AGENT_REGISTRY: list[dict[str, Any]] = [
    {
        "agent_name": "financial_security",
        "slug": "financial",
        "label": "Financial Security",
        "description": "Ordered playbook: ingest, normalize, risk patterns, recommendations, watchlist.",
        "run_entrypoint": "domain.agents.financial_security_agent:run_financial_security_playbook",
        "supports_dry_run": True,
    },
    {
        "agent_name": "graph_drift",
        "slug": "drift",
        "label": "Graph Drift",
        "description": "Multi-metric embedding drift (centroid, MMD, KS); root-cause and drift_warning risk_signal.",
        "run_entrypoint": "domain.agents.graph_drift_agent:run_graph_drift_agent",
        "supports_dry_run": True,
    },
    {
        "agent_name": "evidence_narrative",
        "slug": "narrative",
        "label": "Evidence Narrative",
        "description": "Evidence-grounded narrative for open signals; redaction-aware.",
        "run_entrypoint": "domain.agents.evidence_narrative_agent:run_evidence_narrative_agent",
        "supports_dry_run": True,
    },
    {
        "agent_name": "ring_discovery",
        "slug": "ring",
        "label": "Ring Discovery",
        "description": "Interaction graph clustering (Neo4j GDS or NetworkX); ring_candidate risk_signals.",
        "run_entrypoint": "domain.agents.ring_discovery_agent:run_ring_discovery_agent",
        "supports_dry_run": True,
    },
    {
        "agent_name": "continual_calibration",
        "slug": "calibration",
        "label": "Continual Calibration",
        "description": "Platt scaling / conformal threshold from feedback; calibration report.",
        "run_entrypoint": "domain.agents.continual_calibration_agent:run_continual_calibration_agent",
        "supports_dry_run": True,
    },
    {
        "agent_name": "synthetic_redteam",
        "slug": "redteam",
        "label": "Synthetic Red-Team",
        "description": "Scenario DSL + regression: similar incidents, centroid watchlist, evidence subgraph.",
        "run_entrypoint": "domain.agents.synthetic_redteam_agent:run_synthetic_redteam_agent",
        "supports_dry_run": True,
    },
]


def get_known_agent_names() -> tuple[str, ...]:
    return tuple(a["agent_name"] for a in AGENT_REGISTRY)


def get_agent_by_name(name: str) -> dict[str, Any] | None:
    for a in AGENT_REGISTRY:
        if a["agent_name"] == name:
            return a
    return None


def get_agent_by_slug(slug: str) -> dict[str, Any] | None:
    for a in AGENT_REGISTRY:
        if a["slug"] == slug:
            return a
    return None


def slug_to_agent_name(slug: str) -> str | None:
    a = get_agent_by_slug(slug)
    return a["agent_name"] if a else None
