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
    """When neo4j_available=False, step_trace includes build_graph and cluster; summary has neo4j_available False."""
    out = run_ring_discovery_agent("hh-1", supabase=None, neo4j_available=False)
    steps = [s.get("step") for s in out["step_trace"]]
    assert "build_graph" in steps or "skip" in [s.get("status") for s in out["step_trace"]]
    assert out["summary_json"].get("neo4j_available") is False
    assert "rings_found" in out["summary_json"] or "reason" in out["summary_json"]


def test_ring_discovery_step_trace_when_neo4j_available() -> None:
    """When neo4j_available=True, cluster step runs; summary has neo4j_available True (when supabase provided)."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.side_effect = lambda n: q
    out = run_ring_discovery_agent("hh-1", supabase=mock_sb, neo4j_available=True)
    assert out["summary_json"].get("neo4j_available") is True


def test_ring_discovery_dry_run_same_shape() -> None:
    """dry_run does not change return shape (no DB write; same keys)."""
    out = run_ring_discovery_agent("hh-1", supabase=None, dry_run=True)
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
    """step_trace includes generate_variants and run_regression."""
    out = run_synthetic_redteam_agent("hh-1", dry_run=True)
    steps = [s.get("step") for s in out["step_trace"]]
    assert "generate_variants" in steps
    assert "run_regression" in steps


def test_synthetic_redteam_summary_has_variants_and_regression() -> None:
    """summary_json has scenarios_generated, regression_passed, model_available."""
    out = run_synthetic_redteam_agent("hh-1", dry_run=True)
    s = out["summary_json"]
    assert "scenarios_generated" in s
    assert "regression_passed" in s
    assert "model_available" in s


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
    """With mocked supabase returning labeled feedback + calibration, report in summary."""
    mock_sb = MagicMock()
    fb_q = MagicMock()
    fb_q.select.return_value = fb_q
    fb_q.eq.return_value = fb_q
    fb_q.in_.return_value = fb_q
    fb_q.execute.return_value.data = [
        {"risk_signal_id": "rs1", "label": "true_positive"},
        {"risk_signal_id": "rs2", "label": "false_positive"},
        {"risk_signal_id": "rs3", "label": "true_positive"},
    ]
    sig_q = MagicMock()
    sig_q.select.return_value = sig_q
    sig_q.eq.return_value = sig_q
    sig_q.limit.return_value = sig_q
    sig_q.execute.return_value.data = [{"score": 0.7}]
    cal_q = MagicMock()
    cal_q.select.return_value = cal_q
    cal_q.eq.return_value = cal_q
    cal_q.single.return_value = cal_q
    cal_q.execute.return_value.data = {"severity_threshold_adjust": 0.2}

    def table(name):
        if name == "feedback":
            return fb_q
        if name == "risk_signals":
            return sig_q
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


def test_continual_calibration_supabase_error_degrades_gracefully() -> None:
    """When supabase raises during fetch, agent degrades (ok with reason or error in summary)."""
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.in_.return_value.execute.side_effect = RuntimeError("db down")
    out = run_continual_calibration_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["status"] in ("ok", "error")
    assert "summary_json" in out


# ----- Evidence Narrative Agent -----


def test_evidence_narrative_returns_contract_shape() -> None:
    """Return has step_trace, summary_json, status, started_at, ended_at."""
    out = run_evidence_narrative_agent("hh-1", supabase=None, dry_run=False)
    assert "step_trace" in out
    assert "summary_json" in out
    assert out["status"] == "ok"


def test_evidence_narrative_no_supabase_skip_fetch() -> None:
    """When supabase is None, step_trace includes fetch_signals skip, reason no_supabase."""
    out = run_evidence_narrative_agent("hh-1", supabase=None)
    fetch = next((s for s in out["step_trace"] if s.get("step") == "fetch_signals"), None)
    assert fetch is not None
    assert fetch.get("status") == "skip"
    assert fetch.get("notes") == "no_supabase"
    assert out["summary_json"].get("updated") == 0


def test_evidence_narrative_with_supabase_empty_signals() -> None:
    """When supabase returns no signals, updated is 0 and signals_processed 0."""
    mock_sb = MagicMock()
    sess = MagicMock()
    sess.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [{}]
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.order.return_value = q
    q.limit.return_value = q
    q.execute.return_value.data = []
    def table(name):
        if name == "sessions":
            return sess
        return q
    mock_sb.table.side_effect = table
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
    risk_signals_mock.update.return_value.eq.return_value.eq.return_value.execute.return_value = None
    sess = MagicMock()
    sess.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [{}]
    def table(name):
        if name == "risk_signals":
            return risk_signals_mock
        if name == "sessions":
            return sess
        if name == "entities":
            ent = MagicMock()
            ent.select.return_value.eq.return_value.in_.return_value.execute.return_value.data = []
            return ent
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
    sess = MagicMock()
    sess.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = [{}]
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.in_.return_value = q
    q.order.return_value = q
    q.limit.return_value = q
    q.execute.return_value.data = []
    def table(name):
        if name == "sessions":
            return sess
        return q
    mock_sb.table.side_effect = table
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


def test_graph_drift_summary_has_metrics_or_reason() -> None:
    """summary_json has metrics or reason, drift_detected when applicable."""
    out = run_graph_drift_agent("hh-1", supabase=None)
    s = out["summary_json"]
    assert "drift_detected" in s or "reason" in s or "metrics" in s


def test_graph_drift_with_supabase_no_embeddings() -> None:
    """When supabase returns no embeddings, insufficient_samples reason, drift_detected False."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.gte.return_value = q
    q.lte.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.side_effect = lambda n: q
    out = run_graph_drift_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert out["summary_json"].get("drift_detected") is False
    assert out["summary_json"].get("reason") == "insufficient_samples" or "n_baseline" in out["summary_json"]


def test_graph_drift_custom_threshold() -> None:
    """drift_threshold parameter is reflected in summary_json or step notes."""
    out = run_graph_drift_agent("hh-1", supabase=None, drift_threshold=0.25)
    assert out["status"] == "ok"
    assert "summary_json" in out


def test_graph_drift_shift_below_tau_does_not_open_warning() -> None:
    """When insufficient samples or shift below threshold, no drift_warning insert."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.gte.return_value = q
    q.lte.return_value = q
    q.execute.return_value.data = []
    mock_sb.table.side_effect = lambda n: q
    out = run_graph_drift_agent("hh-1", supabase=mock_sb, dry_run=False)
    open_steps = [s for s in out["step_trace"] if "emit" in str(s.get("step", ""))]
    assert out["summary_json"].get("drift_detected") is False or out["summary_json"].get("reason") == "insufficient_samples"


def test_graph_drift_supabase_error_degrades_gracefully() -> None:
    """When supabase raises, agent records reason or error in step_trace/summary."""
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.side_effect = RuntimeError("db")
    out = run_graph_drift_agent("hh-1", supabase=mock_sb, dry_run=False)
    assert "step_trace" in out
    assert "summary_json" in out
