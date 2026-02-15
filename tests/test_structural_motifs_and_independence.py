"""Tests for structural motifs and independence metadata in explanation."""
from __future__ import annotations


def test_structural_motifs_exposed_in_explanation() -> None:
    """Financial agent exposes structural_motifs in each risk_signal explanation."""
    from domain.agents.financial_security_agent import run_financial_security_playbook, get_demo_events

    result = run_financial_security_playbook(
        household_id="test",
        time_window_days=7,
        consent_state={"share_with_caregiver": True},
        ingested_events=get_demo_events(),
        supabase=None,
        dry_run=True,
    )
    for sig in result.get("risk_signals", []):
        expl = sig.get("explanation") or {}
        assert "structural_motifs" in expl
        assert isinstance(expl["structural_motifs"], list)


def test_independence_metadata_in_explanation() -> None:
    """Explanation includes independence fields (cluster_id, bridges_independent_sets, independent_set_size) when present on entity."""
    from domain.agents.financial_security_agent import run_financial_security_playbook, get_demo_events

    result = run_financial_security_playbook(
        household_id="test",
        time_window_days=7,
        consent_state={"share_with_caregiver": True},
        ingested_events=get_demo_events(),
        supabase=None,
        dry_run=True,
    )
    for sig in result.get("risk_signals", []):
        expl = sig.get("explanation") or {}
        assert "independence" in expl
        assert isinstance(expl["independence"], dict)
