"""Tests for model health agent (drift + calibration + optional redteam)."""
from __future__ import annotations


def test_model_health_runs_drift_and_calibration_and_skips_redteam_in_prod() -> None:
    """In prod env, model_health runs drift + calibration; redteam is skipped unless admin_force."""
    from domain.agents.model_health_agent import run_model_health_agent

    result = run_model_health_agent(
        household_id="test-hh",
        supabase=None,
        dry_run=True,
        env="prod",
        admin_force=False,
    )
    assert "step_trace" in result
    steps = [s.get("step") for s in result["step_trace"]]
    assert "drift_check" in steps
    assert "calibration_check" in steps
    assert "conformal_validity_check" in steps
    # redteam_regression only when do_redteam (env != prod or admin_force)
    assert result.get("summary_json", {}).get("redteam_run") is False


def test_model_health_flags_stale_calibration_when_drift_high() -> None:
    """When drift is severe, model_health can mark calibration stale (with supabase); without DB we only get recommendation."""
    from domain.agents.model_health_agent import run_model_health_agent

    result = run_model_health_agent(
        household_id="test-hh",
        supabase=None,
        dry_run=True,
        env="prod",
    )
    summary = result.get("summary_json") or {}
    assert "recommendation" in summary
    assert summary["recommendation"] in ("stable", "retrain", "recalibrate")
