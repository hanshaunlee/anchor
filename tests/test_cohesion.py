"""Cohesion tests: no placeholder outputs, consent redaction, drift computation."""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from api.schemas import RiskScoringResponse


def test_score_risk_no_checkpoint_returns_no_scores_no_embeddings() -> None:
    """When no model checkpoint exists: model_available=False, scores=[], no embeddings."""
    from domain.risk_scoring_service import score_risk
    from pathlib import Path

    out = score_risk(
        "hh1",
        sessions=[{"id": "s1", "started_at": 0}],
        utterances=[],
        entities=[{"id": "e1", "entity_type": "topic", "canonical": "x"}],
        mentions=[],
        relationships=[],
        checkpoint_path=Path("/nonexistent/checkpoint.pt"),
    )
    assert out.model_available is False
    assert out.scores == []
    assert out.model_meta is None


def test_similar_incidents_unavailable_when_no_embedding() -> None:
    """When no embedding row: similar incidents returns available=false, reason=model_not_run."""
    from domain.explain_service import get_similar_incidents
    from api.schemas import SimilarIncidentsResponse

    sb = MagicMock()
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.limit.return_value = q
    q.gte.return_value = q
    q.execute.return_value.data = []
    sb.table.return_value = q

    result = get_similar_incidents(uuid4(), "hh1", sb, top_k=5)
    assert isinstance(result, SimilarIncidentsResponse)
    assert result.available is False
    assert result.reason == "model_not_run"
    assert result.similar == []


def test_risk_signal_detail_redacts_when_consent_disallows() -> None:
    """When consent_allows_share_text=False, explanation is redacted and subgraph labels stripped."""
    from domain.risk_service import get_risk_signal_detail

    sig_id = uuid4()
    hh_id = str(uuid4())
    sb = MagicMock()
    row = {
        "id": str(sig_id),
        "household_id": hh_id,
        "ts": "2024-01-01T12:00:00Z",
        "signal_type": "relational_anomaly",
        "severity": 3,
        "score": 0.7,
        "status": "open",
        "explanation": {
            "summary": "Sensitive text here",
            "timeline_snippet": [{"text_preview": "User said something"}],
            "model_available": False,
        },
        "recommended_action": {},
    }
    q = MagicMock()
    q.select.return_value = q
    q.eq.return_value = q
    q.single.return_value = q
    q.execute.return_value.data = row
    sb.table.return_value = q

    detail = get_risk_signal_detail(
        sig_id,
        hh_id,
        sb,
        consent_allows_share_text=False,
    )
    assert detail is not None
    assert detail.explanation.get("redacted") is True
    assert "redaction_reason" in detail.explanation
    if detail.explanation.get("timeline_snippet"):
        for item in detail.explanation["timeline_snippet"]:
            if isinstance(item, dict) and "text_preview" in item:
                assert item["text_preview"] == "[Redacted due to consent]"
    if detail.subgraph and detail.subgraph.nodes:
        for node in detail.subgraph.nodes:
            assert node.label is None


def test_graph_drift_compute_shift_range() -> None:
    """_compute_shift returns value in [0, 2]; 0 when identical, 2 when opposite."""
    from domain.agents.graph_drift_agent import _compute_shift

    emb = [1.0, 0.0, 0.0]
    same = _compute_shift([emb, emb], [emb, emb])
    assert same == 0.0

    opposite = [-1.0, 0.0, 0.0]
    shift_opposite = _compute_shift([emb], [opposite])
    assert shift_opposite >= 1.5  # 1 - cos(180°) = 2