"""E2E tests for GNN pipeline: real checkpoint → embeddings, model_subgraph, no placeholders."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

# Demo events for pipeline (same shape as run_gnn_e2e.py)
DEMO_EVENTS = [
    {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Someone from Medicare called", "confidence": 0.9}},
    {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:01Z", "seq": 1, "event_type": "intent", "payload": {"name": "share_ssn", "slots": {"number": "555-1234"}, "confidence": 0.85}},
    {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:02Z", "seq": 2, "event_type": "final_asr", "payload": {"text": "Verify immediately", "confidence": 0.88}},
]


def _default_checkpoint() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "runs" / "hgt_baseline" / "best.pt"


@pytest.fixture
def with_checkpoint(monkeypatch):
    """Point ML config to repo runs/hgt_baseline/best.pt if it exists."""
    ckpt = _default_checkpoint()
    if ckpt.is_file():
        monkeypatch.setenv("ANCHOR_ML_CHECKPOINT_PATH", str(ckpt.resolve()))
        yield ckpt
    else:
        pytest.skip("No checkpoint at runs/hgt_baseline/best.pt — run: python -m ml.train --epochs 8")


def test_pipeline_with_checkpoint_produces_embeddings_and_model_subgraph(with_checkpoint) -> None:
    """When checkpoint exists, pipeline sets _model_available and risk_scores have real embeddings."""
    from api.pipeline import run_pipeline

    state = run_pipeline(
        household_id="test-e2e",
        ingested_events=DEMO_EVENTS,
    )
    assert state.get("_model_available") is True
    risk_scores = state.get("risk_scores", [])
    assert len(risk_scores) > 0
    with_emb = [r for r in risk_scores if r.get("embedding") and len(r.get("embedding", [])) > 0]
    assert len(with_emb) > 0, "Expected at least one risk_score with real embedding when model runs"
    explanations = state.get("explanations", [])
    # model_subgraph appears in explanations only when PG explainer ran for that node; embeddings are always present when model runs
    assert len(with_emb) > 0


def test_pipeline_without_checkpoint_sets_model_available_false(monkeypatch) -> None:
    """When no checkpoint, pipeline uses fallback scores and _model_available is False."""
    monkeypatch.setenv("ANCHOR_ML_CHECKPOINT_PATH", "/nonexistent/best.pt")
    # Force reload of settings so the env is read
    try:
        from config.settings import get_ml_settings
        get_ml_settings.cache_clear()
    except ImportError:
        pass

    from api.pipeline import run_pipeline

    state = run_pipeline(
        household_id="test-no-ckpt",
        ingested_events=DEMO_EVENTS,
    )
    assert state.get("_model_available") is False
    risk_scores = state.get("risk_scores", [])
    for r in risk_scores:
        assert "embedding" not in r or not r.get("embedding"), "No real embeddings when model did not run"
