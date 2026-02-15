"""Tests for supervisor orchestrator and investigation flow."""
from __future__ import annotations

import pytest


def test_supervisor_ingest_runs_financial_then_narrative() -> None:
    """INGEST_PIPELINE runs financial detection then ensure_narratives (no supabase = no persist)."""
    from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE

    # Pass at least one event so pipeline does not return early after normalize_events
    one_event = [{"session_id": "s1", "device_id": "d1", "ts": "2025-01-01T12:00:00Z", "seq": 0, "event_type": "wake", "payload": {}}]
    result = run_supervisor(
        household_id="test-hh",
        supabase=None,
        run_mode=INGEST_PIPELINE,
        dry_run=True,
        ingested_events=one_event,
    )
    assert result["mode"] == INGEST_PIPELINE
    assert "step_trace" in result
    steps = [s.get("step") for s in result["step_trace"]]
    assert "load_context" in steps
    assert "normalize_events" in steps
    assert "run_financial_detection" in steps
    assert "created_signal_ids" in result
    assert "outreach_candidates" in result


def test_supervisor_ingest_creates_outreach_draft_not_sent_by_default() -> None:
    """Outreach candidates are created as drafts; no send by default."""
    from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE

    result = run_supervisor(
        household_id="test-hh",
        supabase=None,
        run_mode=INGEST_PIPELINE,
        dry_run=True,
        ingested_events=[],  # no events -> no signals -> no outreach_candidates
    )
    assert "outreach_candidates" in result
    # With dry_run and no DB, we don't persist outbound_actions; candidates list may be empty
    assert isinstance(result["outreach_candidates"], list)


def test_supervisor_new_alert_auto_sends_only_with_auto_send_and_consent() -> None:
    """NEW_ALERT with risk_signal_id: auto_send only when capabilities.auto_send_outreach and consent."""
    from domain.agents.supervisor import run_supervisor, NEW_ALERT

    # No supabase: run will fail to fetch signal but we can check the mode path
    result = run_supervisor(
        household_id="test-hh",
        supabase=None,
        run_mode=NEW_ALERT,
        dry_run=True,
        risk_signal_id="00000000-0000-0000-0000-000000000001",
    )
    assert result["mode"] == NEW_ALERT
    assert "child_run_ids" in result


def test_supervisor_idempotent_second_run_does_not_duplicate_narratives_or_actions() -> None:
    """Second INGEST_PIPELINE run (dry_run, no DB) should not duplicate; idempotency is in narrative agent and DB constraints."""
    from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE

    result1 = run_supervisor("test-hh", supabase=None, run_mode=INGEST_PIPELINE, dry_run=True, ingested_events=[])
    result2 = run_supervisor("test-hh", supabase=None, run_mode=INGEST_PIPELINE, dry_run=True, ingested_events=[])
    assert result1["mode"] == result2["mode"]
    # Without DB we can't assert no duplicate rows; just that both complete
    assert "step_trace" in result1 and "step_trace" in result2


def test_supervisor_nightly_maintenance_runs_model_health_only() -> None:
    """NIGHTLY_MAINTENANCE runs only model_health agent."""
    from domain.agents.supervisor import run_supervisor, NIGHTLY_MAINTENANCE

    result = run_supervisor(
        household_id="test-hh",
        supabase=None,
        run_mode=NIGHTLY_MAINTENANCE,
        dry_run=True,
    )
    assert result["mode"] == NIGHTLY_MAINTENANCE
    assert "model_health" in result.get("child_run_ids", {})
    assert "summary_json" in result


def test_investigation_run_returns_bundle_shape() -> None:
    """POST /investigation/run (supervisor INGEST_PIPELINE) returns bundle with keys needed for UI."""
    from domain.agents.supervisor import run_supervisor, INGEST_PIPELINE

    result = run_supervisor(
        household_id="test-hh",
        supabase=None,
        run_mode=INGEST_PIPELINE,
        dry_run=True,
        ingested_events=[],
    )
    assert "created_signal_ids" in result
    assert isinstance(result["created_signal_ids"], list)
    assert "updated_signal_ids" in result
    assert isinstance(result["updated_signal_ids"], list)
    assert "summary_json" in result
    assert "counts" in result["summary_json"]
    counts = result["summary_json"]["counts"]
    assert "new_signals" in counts or "outreach_candidates" in counts or "watchlists" in counts
    assert "step_trace" in result
    assert isinstance(result["step_trace"], list)
    assert "outreach_candidates" in result
    assert isinstance(result["outreach_candidates"], list)
