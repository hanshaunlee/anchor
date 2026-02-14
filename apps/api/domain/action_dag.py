"""
Deterministic Action DAG builder for incident response.
Nodes and edges are derived from capabilities and signal — not purely LLM; auditable.
"""
from __future__ import annotations

from typing import Any

# Task type identifiers (match DB enum action_task_type where applicable)
NODE_VERIFY_WITH_ELDER = "verify_with_elder"
NODE_CALL_BANK = "call_bank"
NODE_EMAIL_BANK = "email_bank"
NODE_ENABLE_ALERTS = "enable_alerts"
NODE_LOCK_CARD = "lock_card"
NODE_DEVICE_HIGH_RISK_MODE = "device_high_risk_mode_push"
NODE_NOTIFY_CAREGIVER = "notify_caregiver"
NODE_FILE_REPORT = "file_report"
NODE_FREEZE_CARD_INSTRUCTION = "freeze_card_instruction"
NODE_CHANGE_PASSWORD_INSTRUCTION = "change_password_instruction"


def build_action_graph(
    signal: dict[str, Any],
    capabilities: dict[str, Any],
    consent_allow_outbound: bool,
) -> dict[str, Any]:
    """
    Build a deterministic DAG: nodes (task_type, status, why), edges (dependencies).
    Only includes nodes that are applicable; capability-gated.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    bank_caps = capabilities.get("bank_control_capabilities") or {}
    device_push = capabilities.get("device_policy_push_enabled", True)
    notify_enabled = capabilities.get("notify_sms_enabled") or capabilities.get("notify_email_enabled")

    # 1) VERIFY_WITH_ELDER — always first (no deps)
    nodes.append({
        "id": "verify_with_elder",
        "task_type": "verify_with_elder",
        "status": "ready",
        "why": "Confirm with elder before taking external actions.",
    })
    # 2) NOTIFY_CAREGIVER — if notify enabled and consent
    if notify_enabled and consent_allow_outbound:
        nodes.append({
            "id": "notify_caregiver",
            "task_type": "notify_caregiver",
            "status": "ready",
            "why": "Notify caregiver; consent and notify capability enabled.",
        })
        edges.append({"from": "verify_with_elder", "to": "notify_caregiver"})
    else:
        nodes.append({
            "id": "notify_caregiver",
            "task_type": "notify_caregiver",
            "status": "blocked",
            "why": "Notify disabled or consent disallows outbound contact.",
        })
        edges.append({"from": "verify_with_elder", "to": "notify_caregiver"})

    # 3) DEVICE_HIGH_RISK_MODE — if device_policy_push
    if device_push:
        nodes.append({
            "id": "device_high_risk_mode_push",
            "task_type": "device_high_risk_mode_push",
            "status": "ready",
            "why": "Device policy push enabled; device will enact high-risk mode.",
        })
        edges.append({"from": "verify_with_elder", "to": "device_high_risk_mode_push"})
    else:
        nodes.append({
            "id": "device_high_risk_mode_push",
            "task_type": "device_high_risk_mode_push",
            "status": "blocked",
            "why": "Device policy push not enabled.",
        })

    # 4) CALL_BANK / EMAIL_BANK — always present (guided; caregiver executes)
    nodes.append({
        "id": "call_bank",
        "task_type": "call_bank",
        "status": "ready",
        "why": "Call bank hotline with script; no auto-dial.",
    })
    nodes.append({
        "id": "email_bank",
        "task_type": "email_bank",
        "status": "ready",
        "why": "Email bank with case file; script provided.",
    })
    edges.append({"from": "verify_with_elder", "to": "call_bank"})
    edges.append({"from": "verify_with_elder", "to": "email_bank"})

    # 5) ENABLE_ALERTS — only if capability true
    if bank_caps.get("enable_alerts", True):
        nodes.append({
            "id": "enable_alerts",
            "task_type": "enable_alerts",
            "status": "ready",
            "why": "Bank connector supports enable_alerts.",
        })
        edges.append({"from": "call_bank", "to": "enable_alerts"})
    else:
        nodes.append({
            "id": "enable_alerts",
            "task_type": "enable_alerts",
            "status": "blocked",
            "why": "Bank integration does not support enable_alerts; use call script.",
        })

    # 6) LOCK_CARD — only if capability true
    if bank_caps.get("lock_card", False):
        nodes.append({
            "id": "lock_card",
            "task_type": "lock_card",
            "status": "ready",
            "why": "Bank connector supports lock card.",
        })
        edges.append({"from": "call_bank", "to": "lock_card"})
    else:
        nodes.append({
            "id": "freeze_card_instruction",
            "task_type": "freeze_card_instruction",
            "status": "ready",
            "why": "Bank does not support lock card via API; follow call script to freeze.",
        })
        edges.append({"from": "call_bank", "to": "freeze_card_instruction"})

    # 7) FILE_REPORT — always (prefilled template; no auto submission)
    nodes.append({
        "id": "file_report",
        "task_type": "file_report",
        "status": "ready",
        "why": "Prefilled report template for caregiver to submit if needed.",
    })
    edges.append({"from": "verify_with_elder", "to": "file_report"})

    # 8) CHANGE_PASSWORD_INSTRUCTION — recommended
    nodes.append({
        "id": "change_password_instruction",
        "task_type": "change_password_instruction",
        "status": "ready",
        "why": "Recommend password/2FA change; instruction only.",
    })
    edges.append({"from": "verify_with_elder", "to": "change_password_instruction"})

    return {"nodes": nodes, "edges": edges}
