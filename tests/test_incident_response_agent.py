"""Tests for Incident Response Agent: playbook DAG, incident packet, consent, capabilities."""
from unittest.mock import MagicMock

import pytest

from domain.action_dag import build_action_graph
from domain.agents.incident_response_agent import (
    _build_incident_packet,
    run_incident_response_agent,
)


def test_incident_packet_evidence_refs_only() -> None:
    """Incident packet includes only evidence-backed fields (entity_ids, event_ids)."""
    risk_signal = {
        "id": "rs1",
        "signal_type": "possible_scam",
        "explanation": {
            "timeline_snippet": [{"ts": "2024-01-01", "text": "call"}],
            "motif_tags": ["urgency"],
            "entity_ids": ["e1", "e2"],
            "event_ids": ["ev1"],
        },
    }
    packet = _build_incident_packet(risk_signal, [], ["e1", "e2"], ["ev1"])
    assert "evidence_refs" in packet
    assert "entity_ids" in packet
    assert "event_ids" in packet
    assert packet["entity_ids"] == ["e1", "e2"]
    assert packet["event_ids"] == ["ev1"]
    refs = packet["evidence_refs"]
    assert any("entity_id" in str(r) or r.get("entity_id") for r in refs) or any(
        "event_id" in str(r) or r.get("event_id") for r in refs
    )


def test_run_incident_response_agent_dry_run() -> None:
    """Dry run returns no playbook_id; step_trace and summary present."""
    mock_sb = MagicMock()
    risk_signal = {
        "id": "risk-signal-demo",
        "household_id": "household-demo",
        "signal_type": "possible_scam",
        "severity": 3,
        "explanation": {"entity_ids": [], "event_ids": [], "timeline_snippet": []},
        "recommended_action": {},
    }
    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.single.return_value = t
        t.upsert.return_value.execute.return_value.data = []
        t.execute.return_value.data = risk_signal if name == "risk_signals" else []
        if name == "sessions":
            t.execute.return_value.data = [{"consent_state": {}}]
        if name == "household_capabilities":
            t.execute.return_value.data = []
        if name == "household_consent_defaults":
            t.execute.return_value.data = []
        return t
    mock_sb.table.side_effect = table
    result = run_incident_response_agent(
        "household-demo",
        "risk-signal-demo",
        supabase=mock_sb,
        dry_run=True,
    )
    assert result.get("status") == "ok"
    assert result.get("playbook_id") is None
    assert result.get("incident_packet_id") is None
    assert "step_trace" in result
    assert "summary_json" in result
    steps = [s["step"] for s in result["step_trace"]]
    assert "load_incident" in steps
    assert "build_action_dag" in steps


def test_run_incident_response_agent_raises_when_signal_missing() -> None:
    """When risk_signal not found, agent raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        run_incident_response_agent(
            "hh1",
            "nonexistent-signal",
            supabase=None,  # no DB -> _fetch_risk_signal returns None
            dry_run=False,
        )


def test_incident_response_consent_disallows_outbound_tasks_remain_ready() -> None:
    """When consent disallows outbound, notify is blocked in DAG; call_bank remains ready."""
    capabilities = {
        "notify_sms_enabled": True,
        "notify_email_enabled": True,
        "device_policy_push_enabled": False,
        "bank_control_capabilities": {"enable_alerts": True, "lock_card": False},
    }
    graph = build_action_graph({}, capabilities, consent_allow_outbound=False)
    notify = next((n for n in graph["nodes"] if n["task_type"] == "notify_caregiver"), None)
    assert notify and notify["status"] == "blocked"
    call_bank = next((n for n in graph["nodes"] if n["task_type"] == "call_bank"), None)
    assert call_bank and call_bank["status"] == "ready"
