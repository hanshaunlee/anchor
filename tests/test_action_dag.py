"""Tests for domain.action_dag: deterministic DAG, capability gating."""
import pytest

from domain.action_dag import build_action_graph


def test_build_action_graph_deterministic_shape() -> None:
    """DAG has predictable nodes and edges; not purely LLM."""
    signal = {"id": "s1", "signal_type": "possible_scam_contact", "severity": 3}
    capabilities = {
        "device_policy_push_enabled": True,
        "notify_sms_enabled": True,
        "notify_email_enabled": False,
        "bank_control_capabilities": {"lock_card": False, "enable_alerts": True},
    }
    graph = build_action_graph(signal, capabilities, consent_allow_outbound=True)
    assert "nodes" in graph
    assert "edges" in graph
    node_ids = {n["id"] for n in graph["nodes"]}
    assert "verify_with_elder" in node_ids
    assert "call_bank" in node_ids
    assert "notify_caregiver" in node_ids
    assert "device_high_risk_mode_push" in node_ids
    assert "file_report" in node_ids
    # lock_card false -> freeze_card_instruction present
    assert "freeze_card_instruction" in node_ids or "lock_card" in node_ids
    assert len(graph["edges"]) >= 1


def test_build_action_graph_lock_card_only_if_capability_true() -> None:
    """lock_card task node only when bank_control_capabilities.lock_card is true."""
    signal = {}
    cap_true = {
        "device_policy_push_enabled": False,
        "bank_control_capabilities": {"lock_card": True, "enable_alerts": True},
    }
    graph_true = build_action_graph(signal, cap_true, consent_allow_outbound=False)
    node_types_true = {n["task_type"] for n in graph_true["nodes"]}
    assert "lock_card" in node_types_true

    cap_false = {
        "device_policy_push_enabled": False,
        "bank_control_capabilities": {"lock_card": False, "enable_alerts": True},
    }
    graph_false = build_action_graph(signal, cap_false, consent_allow_outbound=False)
    node_types_false = {n["task_type"] for n in graph_false["nodes"]}
    assert "freeze_card_instruction" in node_types_false
    assert "lock_card" not in node_types_false


def test_build_action_graph_call_bank_always_ready() -> None:
    """CALL_BANK task appears and is ready even when capability false."""
    signal = {}
    capabilities = {
        "device_policy_push_enabled": False,
        "bank_control_capabilities": {"lock_card": False, "enable_alerts": False},
    }
    graph = build_action_graph(signal, capabilities, consent_allow_outbound=False)
    call_bank = next((n for n in graph["nodes"] if n["task_type"] == "call_bank"), None)
    assert call_bank is not None
    assert call_bank["status"] == "ready"


def test_build_action_graph_consent_disallows_outbound_notify_blocked() -> None:
    """When consent_allow_outbound is False, notify_caregiver is blocked."""
    signal = {}
    capabilities = {
        "device_policy_push_enabled": False,
        "notify_sms_enabled": True,
        "notify_email_enabled": True,
        "bank_control_capabilities": {},
    }
    graph = build_action_graph(signal, capabilities, consent_allow_outbound=False)
    notify = next((n for n in graph["nodes"] if n["task_type"] == "notify_caregiver"), None)
    assert notify is not None
    assert notify["status"] == "blocked"
