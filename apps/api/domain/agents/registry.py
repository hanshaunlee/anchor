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
        "display_name": "Financial Security Agent",
        "description": "Ordered playbook: ingest, normalize, risk patterns, recommendations, watchlist.",
        "run_entrypoint": "domain.agents.financial_security_agent:run_financial_security_playbook",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "watchlists"],
        "ui_sections": ["risk_signals", "watchlists", "timeline"],
    },
    {
        "agent_name": "graph_drift",
        "slug": "drift",
        "label": "Graph Drift",
        "display_name": "Drift + Root Cause + Action Plan",
        "description": "Multi-metric embedding drift (centroid, MMD, KS); root-cause and drift_warning risk_signal.",
        "run_entrypoint": "domain.agents.graph_drift_agent:run_graph_drift_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "summaries"],
        "ui_sections": ["drift_chart", "slices", "prototypes"],
    },
    {
        "agent_name": "evidence_narrative",
        "slug": "narrative",
        "label": "Evidence Narrative",
        "display_name": "Investigation Packager",
        "description": "Evidence-grounded narrative for open signals; redaction-aware.",
        "run_entrypoint": "domain.agents.evidence_narrative_agent:run_evidence_narrative_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "summaries"],
        "ui_sections": ["narrative", "hypotheses", "elder_safe"],
    },
    {
        "agent_name": "ring_discovery",
        "slug": "ring",
        "label": "Ring Discovery",
        "display_name": "Ring + Connector + Escalation",
        "description": "Interaction graph clustering (Neo4j GDS or NetworkX); ring_candidate risk_signals.",
        "run_entrypoint": "domain.agents.ring_discovery_agent:run_ring_discovery_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["rings", "risk_signals", "watchlists"],
        "ui_sections": ["rings", "connectors", "ring_graph"],
    },
    {
        "agent_name": "continual_calibration",
        "slug": "calibration",
        "label": "Continual Calibration",
        "display_name": "Calibration + Policy Update",
        "description": "Platt scaling / conformal threshold from feedback; calibration report.",
        "run_entrypoint": "domain.agents.continual_calibration_agent:run_continual_calibration_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["summaries", "risk_signals", "household_calibration"],
        "ui_sections": ["calibration_chart", "policy_patch"],
    },
    {
        "agent_name": "synthetic_redteam",
        "slug": "redteam",
        "label": "Synthetic Red-Team",
        "display_name": "Scenario Generator + Regression Harness",
        "description": "Scenario DSL + regression: similar incidents, centroid watchlist, evidence subgraph.",
        "run_entrypoint": "domain.agents.synthetic_redteam_agent:run_synthetic_redteam_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "replay_fixture", "summaries"],
        "ui_sections": ["pass_rate", "failing_cases", "replay"],
    },
    {
        "agent_name": "caregiver_outreach",
        "slug": "outreach",
        "label": "Caregiver Outreach",
        "display_name": "Caregiver Escalation & Outreach",
        "description": "Outbound notify/call/email to caregiver; consent-gated; evidence bundle and elder-safe message.",
        "run_entrypoint": "domain.agents.caregiver_outreach_agent:run_caregiver_outreach_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["outbound_actions", "risk_signals"],
        "ui_sections": ["outreach", "sent", "suppressed"],
    },
    {
        "agent_name": "incident_response",
        "slug": "incident-response",
        "label": "Incident Response",
        "display_name": "Incident Response / Account Lockdown",
        "description": "Capability-aware: playbook DAG, incident packet, tasks; notify/device per capability; never overpromise.",
        "run_entrypoint": "domain.agents.incident_response_agent:run_incident_response_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["action_playbooks", "action_tasks", "incident_packets"],
        "ui_sections": ["action_plan", "bank_case_file", "device_high_risk"],
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
