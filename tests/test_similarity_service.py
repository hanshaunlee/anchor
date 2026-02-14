"""Comprehensive tests for domain.similarity_service: _cos_sim and get_similar_incidents."""
from __future__ import annotations

from uuid import uuid4
from unittest.mock import MagicMock

import pytest

from api.schemas import SimilarIncidentsResponse


def test_cos_sim_unit_identical_vectors() -> None:
    """_cos_sim returns 1.0 for identical normalized direction."""
    from domain.similarity_service import _cos_sim
    v = [1.0, 0.0, 0.0]
    assert _cos_sim(v, v) == pytest.approx(1.0)


def test_cos_sim_orthogonal() -> None:
    """_cos_sim returns 0 for orthogonal vectors."""
    from domain.similarity_service import _cos_sim
    assert _cos_sim([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_cos_sim_opposite() -> None:
    """_cos_sim returns -1 for opposite direction."""
    from domain.similarity_service import _cos_sim
    assert _cos_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cos_sim_different_length_returns_zero() -> None:
    """_cos_sim returns 0 when lengths differ."""
    from domain.similarity_service import _cos_sim
    assert _cos_sim([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_get_similar_incidents_no_row_returns_unavailable() -> None:
    """When no embedding row for signal, returns available=False, reason=model_not_run, similar=[]."""
    from domain.similarity_service import get_similar_incidents

    mock_sb = MagicMock()
    emb_q = MagicMock()
    emb_q.select.return_value = emb_q
    emb_q.eq.return_value = emb_q
    emb_q.limit.return_value = emb_q
    emb_q.execute.return_value.data = []
    mock_sb.table.return_value = emb_q

    out = get_similar_incidents(uuid4(), "hh-1", mock_sb, top_k=5)
    assert isinstance(out, SimilarIncidentsResponse)
    assert out.available is False
    assert out.reason == "model_not_run"
    assert out.similar == []


def test_get_similar_incidents_has_embedding_false_returns_unavailable() -> None:
    """When row has has_embedding=false, returns available=False, reason=model_not_run."""
    from domain.similarity_service import get_similar_incidents

    mock_sb = MagicMock()
    emb_q = MagicMock()
    emb_q.select.return_value = emb_q
    emb_q.eq.return_value = emb_q
    emb_q.limit.return_value = emb_q
    emb_q.execute.return_value.data = [{"embedding": None, "has_embedding": False}]
    mock_sb.table.return_value = emb_q

    out = get_similar_incidents(uuid4(), "hh-1", mock_sb)
    assert out.available is False
    assert out.reason == "model_not_run"
    assert out.similar == []


def test_get_similar_incidents_empty_embedding_list_returns_unavailable() -> None:
    """When embedding is [] or null, returns available=False."""
    from domain.similarity_service import get_similar_incidents

    mock_sb = MagicMock()
    emb_q = MagicMock()
    emb_q.select.return_value = emb_q
    emb_q.eq.return_value = emb_q
    emb_q.limit.return_value = emb_q
    emb_q.execute.return_value.data = [{"embedding": [], "has_embedding": True}]
    mock_sb.table.return_value = emb_q

    out = get_similar_incidents(uuid4(), "hh-1", mock_sb)
    assert out.available is False
    assert out.reason == "model_not_run"


def test_get_similar_incidents_single_row_data_not_list() -> None:
    """When execute returns single object (not list), service treats as one row."""
    from domain.similarity_service import get_similar_incidents

    mock_sb = MagicMock()
    emb_q = MagicMock()
    emb_q.select.return_value = emb_q
    emb_q.eq.return_value = emb_q
    emb_q.limit.return_value = emb_q
    emb_q.execute.return_value.data = {"embedding": None, "has_embedding": False}
    mock_sb.table.return_value = emb_q

    out = get_similar_incidents(uuid4(), "hh-1", mock_sb)
    assert out.available is False


def test_get_similar_incidents_window_days_parameter() -> None:
    """get_similar_incidents accepts window_days (used in candidate query)."""
    from domain.similarity_service import get_similar_incidents

    mock_sb = MagicMock()
    emb_q = MagicMock()
    emb_q.select.return_value = emb_q
    emb_q.eq.return_value = emb_q
    emb_q.limit.return_value = emb_q
    emb_q.execute.return_value.data = [{"embedding": [0.1] * 32, "has_embedding": True}]
    cand_q = MagicMock()
    cand_q.select.return_value = cand_q
    cand_q.eq.return_value = cand_q
    cand_q.gte.return_value = cand_q
    cand_q.execute.return_value.data = []
    mock_sb.rpc = MagicMock(side_effect=Exception("no rpc"))
    # First table() = query embedding; second table() = fallback candidates
    mock_sb.table.side_effect = [emb_q, cand_q]

    out = get_similar_incidents(uuid4(), "hh-1", mock_sb, top_k=5, window_days=30)
    assert out.available is True
    assert out.similar == []
