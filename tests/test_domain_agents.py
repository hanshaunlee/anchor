"""Comprehensive tests for domain agents: Ring Discovery, Synthetic Red-Team, Continual Calibration, Evidence Narrative, Graph Drift."""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from domain.agents.ring_discovery_agent import run_ring_discovery_agent
from domain.agents.synthetic_redteam_agent import run_synthetic_redteam_agent
from domain.agents.continual_calibration_agent import run_continual_calibration_agent
from domain.agents.evidence_narrative_agent import run_evidence_narrative_agent
from domain.agents.graph_drift_agent import run_graph_drift_agent, DRIFT_THRESHOLD_TAU


# ----- Ring Discovery Agent -----


def test_ring_discovery_returns_contract_shape() -> None:
    """Return dict has step_trace, summary_json, status, started_at, ended_at."""
    out = run_ring_discovery_agent("hh-1", supabase=None, neo4j_available=False, dry_run=False)
    assert "step_trace" in out
    assert "summary_json" in out
    assert "status" in out
    assert "started_at" in out
    assert "ended_at" in out
    assert isinstance(out["step_trace"], list)
    assert isinstance(out["summary_json"], dict)


def test_ring_discovery_step_trace_when_neo4j_unavailable() -> None:
    """When neo4j_available=False, step_trace includes check_neo4j and gds_similarity skip."""
    out = run_ring_discovery_agent("hh-1", neo4j_available=False)
    steps = [s.get("step") for s in out["step_trace"]]
    assert "check_neo4j" in steps
    assert "gds_similarity" in steps
    gds = next(s for s in out["step_trace"] if s.get("step") == "gds_similarity")
    assert gds.get("status") == "skip"
    assert gds.get("reason") == "neo4j_unavailable"
    assert out["summary_json"].get("neo4j_available") is False
    assert out["summary_json"].get("clusters_found") == 0


def test_ring_discovery_step_trace_when_neo4j_available() -> None:
    """When neo4j_available=True, gds_similarity status is ok; summary has neo4j_available True."""
    out = run_ring_discovery_agent("hh-1", neo4j_available=True)
    gds = next(s for s in out["step_trace"] if s.get("step") == "gds_similarity")
    assert gds.get("status") == "ok"
    assert out["summary_json"].get("neo4j_available") is True


def test_ring_discovery_dry_run_same_shape() -> None:
    """dry_run does not change return shape (no DB write; same keys)."""
    out = run_ring_discovery_agent("hh-1", dry_run=True)
    assert set(out.keys()) >= {"step_trace", "summary_json", "status", "started_at", "ended_at"}


# ----- Synthetic Red-Team Agent -----


def test_synthetic_redteam_returns_contract_shape() -> None:
    """Return dict has step_trace, summary_json, status, started_at, ended_at."""
    out = run_synthetic_redteam_agent("hh-1", dry_run=True)
    assert "step_trace" in out
    assert "summary_json" in out
    assert "status" in out
    assert "started_at" in out
    assert "ended_at" in out


def test_synthetic_redteam_step_trace_contains_steps() -> None:
    """step_trace includes generate_variants, validate_similar_incidents, validate_centroid_watchlists."""
    out = run_synthetic_redteam_agent("hh-1", dry_run=True)
    steps = [s.get("step") for s in out["step_trace"]]
    assert "generate_variants" in steps
    assert "validate_similar_incidents" in steps
    assert "validate_centroid_watchlists" in steps


def test_synthetic_redteam_summary_has_variants_and_regression() -> None:
    """summary_json has variants_generated, dry_run, regression_passed."""
    out = run_synthetic_redteam_agent("hh-1", dry_run=True)
    s = out["summary_json"]
    assert "variants_generated" in s
    assert s.get("dry_run") is True
    assert s.get("regression_passed") is True


def test_synthetic_redteam_dry_run_false_same_keys() -> None:
    """dry_run=False still returns same top-level keys."""
    out = run_synthetic_redteam_agent("hh-1", dry_run=False)
    assert "summary_json" in out and "step_trace" in out


# ----- Continual Calibration Agent -----


def test_continual_calibration_returns_contract_shape_no_supabase() -> None:
    """Without supabase, return has step_trace, summary_json, status, started_at, ended_at."""
    out = run_continual_calibration_agent("hh-1", supabase=None, dry_run=False)
    assert "step_trace" in out
    assert "summary_json" in out
    assert out["status"] == "ok"
    assert "started_at" in out and "ended_at" in out


def test_continual_calibration_with_supabase_mock() -> None:
    """With mocked supabase returning feedback count and calibration, report in summary."""
    mock_sb = MagicMock()
    fb_q = MagicMock()
    fb_q.select.return_value = fb_q
    fb_q.eq.return_value = fb_q
    fb_q.execute.return_value.count = 3
    cal_q = MagicMock()
    cal_q.select.return_value = cal_q
    cal_q.eq.return_value = cal_q
    cal_q.single.return_value = cal_q
    cal_q.execute.return_value.data = {"severity_threshold_adjust": 0.2}

    def table(name):
        if name == "feedback":
            return fb_q
        if name == "household_calibration":
            return cal_q
        return MagicMock()

    mock_sb.table.side_effect = table
    out = run_continual_calibration_agent("hh-1", supabase=mock_sb, dry_run=True)
    assert out["status"] == "ok"
    report = out["summary_json"]
    assert report.get("feedback_count") == 3
    assert report.get("current_adjustment") == 0.2
    assert "household_id" in report


def test_continual_calibration_supabase_error_returns_error_status() -> None:
    """When supabase raises, status is error and summary_json contains error."""
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.side_effect = RuntimeError("db down")
    out = run_continual_calibration_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["status"] == "error"
    assert "error" in out["summary_json"]


# ----- Evidence Narrative Agent -----


def test_evidence_narrative_returns_contract_shape() -> None:
    """Return has step_trace, summary_json, status, started_at, ended_at."""
    out = run_evidence_narrative_agent("hh-1", supabase=None, dry_run=False)
    assert "step_trace" in out
    assert "summary_json" in out
    assert out["status"] == "ok"


def test_evidence_narrative_no_supabase_skip_narrative() -> None:
    """When supabase is None, step_trace includes narrative step with status skip, reason no_supabase."""
    out = run_evidence_narrative_agent("hh-1", supabase=None)
    narrative = next((s for s in out["step_trace"] if s.get("step") == "narrative"), None)
    assert narrative is not None
    assert narrative.get("status") == "skip"
    assert narrative.get("reason") == "no_supabase"
    assert out["summary_json"].get("updated") == 0


def test_evidence_narrative_with_supabase_empty_signals() -> None:
    """When supabase returns no signals, updated is 0 and signals_processed 0."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.order.return_value = q
    q.limit.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.return_value = q
    out = run_evidence_narrative_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["status"] == "ok"
    assert out["summary_json"].get("updated") == 0
    assert out["summary_json"].get("signals_processed") == 0


def test_evidence_narrative_builds_narrative_from_subgraph_and_motifs() -> None:
    """When signal has model_subgraph nodes and motif_tags, narrative step runs and summary updated."""
    mock_sb = MagicMock()
    sig_id = str(uuid4())
    risk_signals_mock = MagicMock()
    risk_signals_mock.select.return_value = risk_signals_mock
    risk_signals_mock.eq.return_value = risk_signals_mock
    risk_signals_mock.order.return_value = risk_signals_mock
    risk_signals_mock.limit.return_value = risk_signals_mock
    risk_signals_mock.execute.return_value.data = [
        {
            "id": sig_id,
            "explanation": {
                "model_subgraph": {"nodes": [{"id": "e1"}], "edges": []},
                "motif_tags": ["urgent_contact", "sensitive_topic"],
            },
        },
    ]
    risk_signals_mock.update.return_value.eq.return_value.execute.return_value = None
    def table(name):
        if name == "risk_signals":
            return risk_signals_mock
        return MagicMock()
    mock_sb.table.side_effect = table
    out = run_evidence_narrative_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["status"] == "ok"
    assert out["summary_json"].get("updated") == 1
    assert out["summary_json"].get("signals_processed") == 1


def test_evidence_narrative_skips_signal_with_no_nodes_no_motifs() -> None:
    """Signals with no model_subgraph nodes and no motif_tags are skipped (no update)."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.order.return_value = q
    q.limit.return_value = q
    q.execute.return_value.data = [{"id": str(uuid4()), "explanation": {}}]
    mock_sb.table.return_value = q
    out = run_evidence_narrative_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["summary_json"].get("updated") == 0
    assert out["summary_json"].get("signals_processed") == 1


def test_evidence_narrative_single_signal_by_id() -> None:
    """When risk_signal_id is provided, agent runs and completes (query scoped to that id in implementation)."""
    mock_sb = MagicMock()
    sig_id = str(uuid4())
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.order.return_value = q
    q.limit.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.return_value = q
    out = run_evidence_narrative_agent("hh-1", risk_signal_id=sig_id, supabase=mock_sb)
    assert out["status"] == "ok"
    assert "fetch_signals" in [s.get("step") for s in out["step_trace"]]
    assert out["summary_json"].get("signals_processed") == 0


# ----- Graph Drift Agent -----


def test_graph_drift_returns_contract_shape() -> None:
    """Return has step_trace, summary_json, status, started_at, ended_at."""
    out = run_graph_drift_agent("hh-1", supabase=None, dry_run=False)
    assert "step_trace" in out
    assert "summary_json" in out
    assert "status" in out
    assert "started_at" in out
    assert "ended_at" in out


def test_graph_drift_summary_has_shift_and_threshold() -> None:
    """summary_json has shift, n_embeddings, threshold, drift_detected."""
    out = run_graph_drift_agent("hh-1", supabase=None)
    s = out["summary_json"]
    assert "shift" in s
    assert "n_embeddings" in s
    assert "threshold" in s
    assert "drift_detected" in s
    assert s["threshold"] == DRIFT_THRESHOLD_TAU


def test_graph_drift_with_supabase_no_embeddings() -> None:
    """When supabase returns no embeddings, shift stays 0, drift_detected False."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.gte.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.return_value = q
    out = run_graph_drift_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["summary_json"]["shift"] == 0.0
    assert out["summary_json"]["drift_detected"] is False


def test_graph_drift_custom_tau() -> None:
    """tau parameter is reflected in summary_json threshold."""
    out = run_graph_drift_agent("hh-1", supabase=None, tau=0.25)
    assert out["summary_json"]["threshold"] == 0.25


def test_graph_drift_shift_below_tau_does_not_open_warning() -> None:
    """When shift <= tau, no drift_warning insert (step_trace has no open_drift_warning or dry_run)."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.gte.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.return_value = q
    out = run_graph_drift_agent("hh-1", supabase=mock_sb, dry_run=False)
    open_steps = [s for s in out["step_trace"] if "open_drift_warning" in str(s.get("step", ""))]
    assert len(open_steps) == 0


def test_graph_drift_supabase_error_returns_error_status() -> None:
    """When supabase raises during fetch, status is error and summary has error key."""
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.gte.return_value.eq.return_value.execute.side_effect = RuntimeError("db")
    out = run_graph_drift_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["status"] == "error"
    assert "error" in out["summary_json"]
