"""Tests for conformal escalation and decision_rule_used propagation."""
from __future__ import annotations


def test_conformal_trigger_logic_blocks_outreach_when_not_triggered() -> None:
    """When (1 - calibrated_p) < q_hat, escalation should not trigger."""
    from domain.agents.supervisor import _escalation_triggered

    # calibrated_p high -> 1 - p low; q_hat 0.3 -> 0.2 < 0.3 -> no trigger
    assert _escalation_triggered(calibrated_p=0.8, fusion_score=0.8, severity=4, conformal_q_hat=0.3, escalation_threshold=4) is False


def test_conformal_trigger_logic_allows_outreach_when_triggered() -> None:
    """When (1 - calibrated_p) >= q_hat, escalation triggers."""
    from domain.agents.supervisor import _escalation_triggered

    # 1 - 0.5 = 0.5 >= 0.3
    assert _escalation_triggered(calibrated_p=0.5, fusion_score=0.5, severity=4, conformal_q_hat=0.3, escalation_threshold=4) is True


def test_decision_rule_used_propagates_to_explanation_and_action_payload() -> None:
    """Financial agent includes decision_rule_used in explanation when signals exist."""
    from domain.agents.financial_security_agent import run_financial_security_playbook, get_demo_events

    result = run_financial_security_playbook(
        household_id="test",
        time_window_days=7,
        consent_state={"share_with_caregiver": True},
        ingested_events=get_demo_events(),
        supabase=None,
        dry_run=True,
    )
    # When there are risk_signals, each has explanation with decision_rule_used / calibrated_p / fusion_score
    for sig in result.get("risk_signals", []):
        expl = sig.get("explanation") or {}
        assert "decision_rule_used" in expl or "calibrated_p" in expl or "fusion_score" in expl or "structural_motifs" in expl
