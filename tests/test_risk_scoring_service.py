"""Comprehensive tests for domain.risk_scoring_service: score_risk contract and edge cases."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from api.schemas import RiskScoringResponse, RiskScoreItem


def test_score_risk_empty_entities_returns_model_available_false() -> None:
    """When entities list is empty, returns model_available=False and scores=[]."""
    from domain.risk_scoring_service import score_risk

    out = score_risk(
        "hh-1",
        sessions=[],
        utterances=[],
        entities=[],
        mentions=[],
        relationships=[],
    )
    assert isinstance(out, RiskScoringResponse)
    assert out.model_available is False
    assert out.scores == []
    assert out.fallback_used is None


def test_score_risk_missing_checkpoint_returns_model_available_false() -> None:
    """When checkpoint path does not exist, returns model_available=False and scores=[]."""
    from domain.risk_scoring_service import score_risk

    out = score_risk(
        "hh-1",
        sessions=[],
        utterances=[],
        entities=[{"id": "e1"}],
        mentions=[],
        relationships=[],
        checkpoint_path=Path("/nonexistent/checkpoint.pt"),
    )
    assert out.model_available is False
    assert out.scores == []


def test_score_risk_accepts_optional_events_and_devices() -> None:
    """score_risk accepts events=[] and devices=[] without error (still returns model_available=False if no checkpoint)."""
    from domain.risk_scoring_service import score_risk

    out = score_risk(
        "hh-1",
        sessions=[],
        utterances=[],
        entities=[{"id": "e1"}],
        mentions=[],
        relationships=[],
        devices=[],
        events=[],
        checkpoint_path=Path("/nonexistent.pt"),
    )
    assert out.model_available is False


def test_score_risk_explanation_score_min_optional() -> None:
    """explanation_score_min can be omitted (uses config default) or passed."""
    from domain.risk_scoring_service import score_risk

    out = score_risk(
        "hh-1",
        sessions=[],
        utterances=[],
        entities=[{"id": "e1"}],
        mentions=[],
        relationships=[],
        checkpoint_path=Path("/nonexistent.pt"),
        explanation_score_min=0.5,
    )
    assert out.model_available is False


def test_risk_scoring_response_schema() -> None:
    """RiskScoringResponse and RiskScoreItem validate expected shapes."""
    item = RiskScoreItem(node_index=0, score=0.7, model_available=False)
    assert item.node_type == "entity"
    assert item.signal_type == "relational_anomaly"
    assert item.embedding is None
    assert item.model_subgraph is None

    resp = RiskScoringResponse(model_available=True, scores=[item])
    assert resp.model_available is True
    assert len(resp.scores) == 1
    assert resp.scores[0].score == 0.7
    assert resp.fallback_used is None


def test_risk_scoring_response_fallback_used_field() -> None:
    """RiskScoringResponse accepts optional fallback_used for rule-only path."""
    resp = RiskScoringResponse(
        model_available=False,
        scores=[],
        fallback_used="rule_only",
    )
    assert resp.fallback_used == "rule_only"


def test_risk_score_item_with_model_subgraph() -> None:
    """RiskScoreItem can have model_subgraph and embedding when model ran."""
    item = RiskScoreItem(
        node_index=0,
        score=0.8,
        model_available=True,
        model_subgraph={"nodes": [{"id": "e1", "score": 0.8}], "edges": []},
        embedding=[0.1, 0.2],
    )
    assert item.model_subgraph is not None
    assert item.model_subgraph["nodes"][0]["score"] == 0.8
    assert item.embedding == [0.1, 0.2]
