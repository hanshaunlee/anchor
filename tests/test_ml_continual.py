"""Tests for ml.continual.finetune_last_layer: load_feedback_batch, apply_threshold_adjustment, finetune_last_layer_stub."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml.continual.finetune_last_layer import (
    load_feedback_batch,
    apply_threshold_adjustment,
    finetune_last_layer_stub,
)


def test_load_feedback_batch_empty() -> None:
    mock_sb = MagicMock()
    mock_sb.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    out = load_feedback_batch(mock_sb, "hh1", since_ts=None)
    assert out == []


def test_load_feedback_batch_with_since() -> None:
    mock_sb = MagicMock()
    q = mock_sb.table.return_value.select.return_value.eq.return_value
    q.gte.return_value = q
    q.execute.return_value.data = [{"risk_signal_id": "rs1", "label": "true_positive", "notes": None}]
    out = load_feedback_batch(mock_sb, "hh1", since_ts="2024-01-01T00:00:00Z")
    assert len(out) == 1
    assert out[0]["risk_signal_id"] == "rs1"


def test_apply_threshold_adjustment() -> None:
    mock_sb = MagicMock()
    apply_threshold_adjustment(mock_sb, "hh1", 0.15)
    mock_sb.table.return_value.upsert.assert_called_once()
    call = mock_sb.table.return_value.upsert.call_args[0][0]
    assert call["household_id"] == "hh1"
    assert call["severity_threshold_adjust"] == 0.15


def test_finetune_last_layer_stub() -> None:
    out = finetune_last_layer_stub(Path("fake.pt"), [{"risk_signal_id": "rs1", "label": "true_positive"}], output_path=None)
    assert out["feedback_count"] == 1
    assert out["status"] == "stub"
    assert out["saved"] is False
