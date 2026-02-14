"""Comprehensive tests for domain.risk_service: list_risk_signals, get_risk_signal_detail, submit_feedback."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from api.schemas import FeedbackSubmit, FeedbackLabel, RiskSignalStatus
from domain.risk_service import list_risk_signals, get_risk_signal_detail, submit_feedback


def test_list_risk_signals_empty_household_returns_empty() -> None:
    """When household_id is empty string, returns signals=[], total=0."""
    mock_sb = MagicMock()
    out = list_risk_signals("", mock_sb)
    assert out.signals == []
    assert out.total == 0
    mock_sb.table.assert_not_called()


def test_list_risk_signals_calls_supabase_with_filters() -> None:
    """list_risk_signals builds query with household_id, optional status and severity_min."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.gte.return_value = q
    q.order.return_value = q
    q.range.return_value = q
    q.execute.return_value.data = []
    q.execute.return_value.count = 0
    mock_sb.table.return_value = q

    list_risk_signals("hh-1", mock_sb, status=RiskSignalStatus.open, severity_min=3, limit=10, offset=0)
    mock_sb.table.assert_called_with("risk_signals")
    q.eq.assert_any_call("household_id", "hh-1")
    q.eq.assert_any_call("status", "open")
    q.gte.assert_called_with("severity", 3)
    q.range.assert_called_with(0, 9)


def test_list_risk_signals_maps_rows_to_risk_signal_cards() -> None:
    """List response maps DB rows to RiskSignalCard with id, ts, signal_type, severity, score, status, summary."""
    mock_sb = MagicMock()
    sig_id = str(uuid4())
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.order.return_value = q
    q.range.return_value = q
    q.execute.return_value.data = [
        {
            "id": sig_id,
            "ts": "2024-01-15T10:00:00+00:00",
            "signal_type": "relational_anomaly",
            "severity": 4,
            "score": 0.8,
            "status": "open",
            "explanation": {"summary": "High risk pattern", "model_available": True},
        },
    ]
    q.execute.return_value.count = 1
    mock_sb.table.return_value = q

    out = list_risk_signals("hh-1", mock_sb)
    assert out.total == 1
    assert len(out.signals) == 1
    assert str(out.signals[0].id) == sig_id
    assert out.signals[0].severity == 4
    assert out.signals[0].score == 0.8
    assert out.signals[0].status == RiskSignalStatus.open
    assert out.signals[0].summary == "High risk pattern"
    assert out.signals[0].model_available is True


def test_get_risk_signal_detail_not_found_returns_none() -> None:
    """When signal not in DB or wrong household, returns None."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.single.return_value = q
    q.execute.return_value.data = None
    mock_sb.table.return_value = q

    out = get_risk_signal_detail(uuid4(), "hh-1", mock_sb)
    assert out is None


def test_get_risk_signal_detail_returns_detail_with_subgraph() -> None:
    """When signal found, returns RiskSignalDetail with explanation and subgraph from model_subgraph."""
    mock_sb = MagicMock()
    sig_id = uuid4()
    hh_id = str(uuid4())
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.single.return_value = q
    q.execute.return_value.data = {
        "id": str(sig_id),
        "household_id": hh_id,
        "ts": "2024-01-15T10:00:00+00:00",
        "signal_type": "relational_anomaly",
        "severity": 4,
        "score": 0.8,
        "status": "open",
        "explanation": {
            "model_subgraph": {"nodes": [{"id": "e1", "type": "entity", "score": 0.8}], "edges": []},
            "model_available": True,
        },
        "recommended_action": {"checklist": ["Review contact"]},
    }
    mock_sb.table.return_value = q

    out = get_risk_signal_detail(sig_id, hh_id, mock_sb)
    assert out is not None
    assert out.id == sig_id
    assert out.explanation.get("model_available") is True
    assert out.subgraph is not None
    assert len(out.subgraph.nodes) == 1
    assert out.subgraph.nodes[0].id == "e1"
    assert out.recommended_action.get("checklist") == ["Review contact"]


def test_get_risk_signal_detail_sets_model_available_false_when_missing() -> None:
    """When explanation has no model_available key, detail sets it to False."""
    mock_sb = MagicMock()
    sig_id = uuid4()
    hh_id = str(uuid4())
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.single.return_value = q
    q.execute.return_value.data = {
        "id": str(sig_id),
        "household_id": hh_id,
        "ts": "2024-01-15T10:00:00+00:00",
        "signal_type": "relational_anomaly",
        "severity": 3,
        "score": 0.5,
        "status": "open",
        "explanation": {"summary": "Rule-based"},
        "recommended_action": {},
    }
    mock_sb.table.return_value = q

    out = get_risk_signal_detail(sig_id, hh_id, mock_sb)
    assert out is not None
    assert out.explanation.get("model_available") is False


def test_submit_feedback_signal_not_found_raises_value_error() -> None:
    """submit_feedback raises ValueError when risk signal not found or not in household."""
    mock_sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.single.return_value = q
    q.execute.return_value.data = None
    mock_sb.table.return_value = q

    with pytest.raises(ValueError, match="not found"):
        submit_feedback(
            uuid4(),
            "hh-1",
            FeedbackSubmit(label=FeedbackLabel.true_positive, notes=None),
            "user-1",
            mock_sb,
        )


def test_submit_feedback_success_inserts_feedback_and_updates_calibration() -> None:
    """submit_feedback inserts feedback row; for true_positive/false_positive updates household_calibration."""
    mock_sb = MagicMock()
    sig_id = uuid4()
    sig_q = MagicMock()
    sig_q.select.return_value = sig_q
    sig_q.eq.return_value = sig_q
    sig_q.single.return_value = sig_q
    sig_q.execute.return_value.data = {"id": str(sig_id)}
    fb_ins = MagicMock()
    fb_ins.insert.return_value.execute.return_value = None
    cal_q = MagicMock()
    cal_q.select.return_value = cal_q
    cal_q.eq.return_value = cal_q
    cal_q.single.return_value = cal_q
    cal_q.execute.return_value.data = {"severity_threshold_adjust": 0.0}
    cal_q.upsert.return_value.execute.return_value = None

    def table(name):
        if name == "risk_signals":
            return sig_q
        if name == "feedback":
            return fb_ins
        if name == "household_calibration":
            return cal_q
        return MagicMock()

    mock_sb.table.side_effect = table
    submit_feedback(
        sig_id,
        "hh-1",
        FeedbackSubmit(label=FeedbackLabel.true_positive, notes=None),
        "user-1",
        mock_sb,
    )
    fb_ins.insert.assert_called_once()
    cal_q.upsert.assert_called_once()
    assert cal_q.upsert.call_args[0][0]["household_id"] == "hh-1"


def test_submit_feedback_unsure_does_not_require_calibration_upsert() -> None:
    """submit_feedback with label unsure still inserts feedback; calibration path may not be hit."""
    mock_sb = MagicMock()
    sig_q = MagicMock()
    sig_q.select.return_value = sig_q
    sig_q.eq.return_value = sig_q
    sig_q.single.return_value = sig_q
    sig_q.execute.return_value.data = {"id": str(uuid4())}
    fb_ins = MagicMock()
    fb_ins.insert.return_value.execute.return_value = None

    def table(name):
        if name == "risk_signals":
            return sig_q
        if name == "feedback":
            return fb_ins
        return MagicMock()

    mock_sb.table.side_effect = table
    submit_feedback(
        uuid4(),
        "hh-1",
        FeedbackSubmit(label=FeedbackLabel.unsure, notes="Need to check"),
        "user-1",
        mock_sb,
    )
    fb_ins.insert.assert_called_once()
