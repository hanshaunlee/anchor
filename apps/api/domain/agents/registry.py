"""
Agent registry: tiering, triggers, visibility, and catalog for product cohesion.
Financial + Narrative are Investigation (externally unified); Outreach is User Action; Drift/Calibration/Redteam are System/DevTools.
"""
from __future__ import annotations

from typing import Any

# --- Tier: where the agent lives in the product ---
AgentTier = str  # Literal["USER_ACTION", "INVESTIGATION", "SYSTEM_MAINTENANCE", "DEVTOOLS"]
USER_ACTION = "USER_ACTION"
INVESTIGATION = "INVESTIGATION"
SYSTEM_MAINTENANCE = "SYSTEM_MAINTENANCE"
DEVTOOLS = "DEVTOOLS"

# --- Trigger: how the agent is invoked ---
AgentTrigger = str  # Literal["MANUAL_UI", "AUTOMATIC", "SCHEDULED", "WEBHOOK", "ADMIN_ONLY"]
MANUAL_UI = "MANUAL_UI"
AUTOMATIC = "AUTOMATIC"
SCHEDULED = "SCHEDULED"
WEBHOOK = "WEBHOOK"
ADMIN_ONLY = "ADMIN_ONLY"

# --- Visibility: where it appears in UI ---
AgentVisibility = str  # Literal["DEFAULT_UI", "ADVANCED_UI", "HIDDEN"]
DEFAULT_UI = "DEFAULT_UI"
ADVANCED_UI = "ADVANCED_UI"
HIDDEN = "HIDDEN"


def _triggers(*args: str) -> list[str]:
    return list(args)


# Slug -> agent metadata (tier, triggers, visibility, capabilities)
AGENT_SPEC: dict[str, dict[str, Any]] = {
    "financial": {
        "agent_name": "financial_security",
        "slug": "financial",
        "label": "Financial Security",
        "display_name": "Financial Security Agent",
        "description": "Ordered playbook: ingest, normalize, risk patterns (calibrated + fusion), recommendations, watchlist.",
        "tier": INVESTIGATION,
        "triggers": _triggers(AUTOMATIC, MANUAL_UI),
        "visibility": DEFAULT_UI,
        "required_roles": ["caregiver", "admin"],
        "consent_requirements": ["share_with_caregiver"],
        "requires_calibrated_model": False,
        "requires_embeddings": False,
        "affects_user_alerts": True,
        "writes_embeddings": True,
        "run_entrypoint": "domain.agents.financial_security_agent:run_financial_security_playbook",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "watchlists"],
        "ui_sections": ["risk_signals", "watchlists", "timeline"],
    },
    "narrative": {
        "agent_name": "evidence_narrative",
        "slug": "narrative",
        "label": "Evidence Narrative",
        "display_name": "Investigation Packager",
        "description": "Evidence-grounded narrative for open signals; redaction-aware. Run automatically as part of Investigation.",
        "tier": INVESTIGATION,
        "triggers": _triggers(AUTOMATIC),
        "visibility": HIDDEN,
        "required_roles": ["caregiver", "admin"],
        "consent_requirements": ["share_with_caregiver"],
        "requires_calibrated_model": False,
        "requires_embeddings": False,
        "affects_user_alerts": True,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.evidence_narrative_agent:run_evidence_narrative_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "summaries"],
        "ui_sections": ["narrative", "hypotheses", "elder_safe"],
    },
    "ring": {
        "agent_name": "ring_discovery",
        "slug": "ring",
        "label": "Ring Discovery",
        "display_name": "Ring + Connector + Escalation",
        "description": "Interaction graph clustering; ring_candidate risk_signals. Conditional in Investigation.",
        "tier": INVESTIGATION,
        "triggers": _triggers(AUTOMATIC, ADMIN_ONLY),
        "visibility": ADVANCED_UI,
        "required_roles": ["caregiver", "admin"],
        "consent_requirements": [],
        "requires_calibrated_model": False,
        "requires_embeddings": False,
        "affects_user_alerts": True,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.ring_discovery_agent:run_ring_discovery_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["rings", "risk_signals", "watchlists"],
        "ui_sections": ["rings", "connectors", "ring_graph"],
    },
    "outreach": {
        "agent_name": "caregiver_outreach",
        "slug": "outreach",
        "label": "Caregiver Outreach",
        "display_name": "Caregiver Escalation & Outreach",
        "description": "Outbound notify/call/email; consent-gated. Action system: preview/send from Alert detail, not 'run agent'.",
        "tier": USER_ACTION,
        "triggers": _triggers(AUTOMATIC, MANUAL_UI),
        "visibility": DEFAULT_UI,
        "required_roles": ["caregiver", "admin"],
        "consent_requirements": ["outbound_contact_ok"],
        "requires_calibrated_model": False,
        "requires_embeddings": False,
        "affects_user_alerts": True,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.caregiver_outreach_agent:run_caregiver_outreach_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["outbound_actions", "risk_signals"],
        "ui_sections": ["outreach", "sent", "suppressed"],
    },
    "supervisor": {
        "agent_name": "supervisor",
        "slug": "supervisor",
        "label": "Investigation",
        "display_name": "Investigation (Supervisor)",
        "description": "Single orchestrated flow: financial detection + narrative + outreach candidates.",
        "tier": INVESTIGATION,
        "triggers": _triggers(AUTOMATIC, MANUAL_UI),
        "visibility": DEFAULT_UI,
        "required_roles": ["caregiver", "admin"],
        "consent_requirements": ["share_with_caregiver"],
        "requires_calibrated_model": False,
        "requires_embeddings": False,
        "affects_user_alerts": True,
        "writes_embeddings": True,
        "run_entrypoint": "domain.agents.supervisor:run_supervisor",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "watchlists", "outreach_candidates"],
        "ui_sections": ["investigation", "trace", "artifacts"],
    },
    "model_health": {
        "agent_name": "model_health",
        "slug": "model_health",
        "label": "Model Health",
        "display_name": "Model Health (Drift + Calibration + Conformal + Redteam)",
        "description": "Unified: drift check, calibration, conformal validity, optional redteam. Scheduled or Admin only.",
        "tier": SYSTEM_MAINTENANCE,
        "triggers": _triggers(SCHEDULED, ADMIN_ONLY),
        "visibility": ADVANCED_UI,
        "required_roles": ["admin"],
        "consent_requirements": [],
        "requires_calibrated_model": False,
        "requires_embeddings": True,
        "affects_user_alerts": False,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.model_health_agent:run_model_health_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["summaries", "household_calibration", "risk_signals"],
        "ui_sections": ["drift_chart", "calibration_chart", "recommendation"],
    },
    "drift": {
        "agent_name": "graph_drift",
        "slug": "drift",
        "label": "Graph Drift",
        "display_name": "Drift + Root Cause + Action Plan",
        "description": "Multi-metric embedding drift (centroid, MMD, KS); drift_warning risk_signal. Part of Model Health.",
        "tier": SYSTEM_MAINTENANCE,
        "triggers": _triggers(ADMIN_ONLY),
        "visibility": ADVANCED_UI,
        "required_roles": ["admin"],
        "consent_requirements": [],
        "requires_calibrated_model": False,
        "requires_embeddings": True,
        "affects_user_alerts": True,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.graph_drift_agent:run_graph_drift_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "summaries"],
        "ui_sections": ["drift_chart", "slices", "prototypes"],
    },
    "calibration": {
        "agent_name": "continual_calibration",
        "slug": "calibration",
        "label": "Continual Calibration",
        "display_name": "Calibration + Policy Update",
        "description": "Platt scaling / split conformal from feedback. Part of Model Health.",
        "tier": SYSTEM_MAINTENANCE,
        "triggers": _triggers(ADMIN_ONLY),
        "visibility": ADVANCED_UI,
        "required_roles": ["admin"],
        "consent_requirements": [],
        "requires_calibrated_model": False,
        "requires_embeddings": False,
        "affects_user_alerts": False,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.continual_calibration_agent:run_continual_calibration_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["summaries", "risk_signals", "household_calibration"],
        "ui_sections": ["calibration_chart", "policy_patch"],
    },
    "redteam": {
        "agent_name": "synthetic_redteam",
        "slug": "redteam",
        "label": "Synthetic Red-Team",
        "display_name": "Scenario Generator + Regression Harness",
        "description": "Scenario DSL + regression. DevTools; not in production or admin_force.",
        "tier": DEVTOOLS,
        "triggers": _triggers(ADMIN_ONLY),
        "visibility": ADVANCED_UI,
        "required_roles": ["admin"],
        "consent_requirements": [],
        "requires_calibrated_model": False,
        "requires_embeddings": True,
        "affects_user_alerts": False,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.synthetic_redteam_agent:run_synthetic_redteam_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["risk_signals", "replay_fixture", "summaries"],
        "ui_sections": ["pass_rate", "failing_cases", "replay"],
    },
    "incident_response": {
        "agent_name": "incident_response",
        "slug": "incident-response",
        "label": "Incident Response",
        "display_name": "Incident Response / Account Lockdown",
        "description": "Capability-aware playbook DAG, incident packet, tasks.",
        "tier": USER_ACTION,
        "triggers": _triggers(MANUAL_UI, ADMIN_ONLY),
        "visibility": ADVANCED_UI,
        "required_roles": ["caregiver", "admin"],
        "consent_requirements": [],
        "requires_calibrated_model": False,
        "requires_embeddings": False,
        "affects_user_alerts": True,
        "writes_embeddings": False,
        "run_entrypoint": "domain.agents.incident_response_agent:run_incident_response_agent",
        "supports_dry_run": True,
        "primary_artifacts": ["action_playbooks", "action_tasks", "incident_packets"],
        "ui_sections": ["action_plan", "bank_case_file", "device_high_risk"],
    },
}

# Backward compatibility: list form for /agents/status (same keys as before)
_LEGACY_KEYS = ("agent_name", "slug", "label", "display_name", "description", "run_entrypoint", "supports_dry_run", "primary_artifacts", "ui_sections")
AGENT_REGISTRY: list[dict[str, Any]] = [
    {k: spec[k] for k in _LEGACY_KEYS if k in spec}
    for spec in AGENT_SPEC.values()
]


def get_known_agent_names() -> tuple[str, ...]:
    return tuple(s["agent_name"] for s in AGENT_SPEC.values())


def get_agent_by_name(name: str) -> dict[str, Any] | None:
    for spec in AGENT_SPEC.values():
        if spec["agent_name"] == name:
            return spec
    return None


def get_agent_by_slug(slug: str) -> dict[str, Any] | None:
    return AGENT_SPEC.get(slug)


def slug_to_agent_name(slug: str) -> str | None:
    a = get_agent_by_slug(slug)
    return a["agent_name"] if a else None


def get_agents_catalog(
    *,
    role: str = "caregiver",
    consent: dict[str, Any] | None = None,
    environment: str = "prod",
    calibration_present: bool = False,
    model_available: bool = False,
) -> list[dict[str, Any]]:
    """
    Return agent catalog entries filtered by role, consent, environment, calibration, model.
    Used by GET /agents/catalog. Each entry includes tier, triggers, visibility, and whether
    the agent is runnable / visible for this context.
    """
    consent = consent or {}
    catalog: list[dict[str, Any]] = []
    for slug, spec in AGENT_SPEC.items():
        entry = dict(spec)
        required_roles = entry.get("required_roles") or []
        consent_reqs = entry.get("consent_requirements") or []
        can_run = role in required_roles if required_roles else True
        for key in consent_reqs:
            if not consent.get(key, True):
                can_run = False
                break
        # Redteam: only in non-prod or when explicitly allowed
        if slug == "redteam" and environment == "prod":
            entry["runnable"] = False
            entry["reason"] = "redteam_disabled_in_production"
        else:
            entry["runnable"] = can_run
            entry["reason"] = None
        if entry.get("requires_calibrated_model") and not calibration_present:
            entry["runnable"] = False
            entry["reason"] = entry.get("reason") or "calibration_required"
        if entry.get("requires_embeddings") and not model_available:
            entry["recommended"] = False  # can still run; may fall back
        else:
            entry["recommended"] = True
        catalog.append(entry)
    return catalog
