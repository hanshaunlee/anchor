"""Tests for risk_signals router: get_risk_signal, get_similar_incidents, submit_feedback (mocked)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user


@pytest.fixture
def client_risk_signals():
    hh_uuid = str(uuid4())
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.limit.return_value = user_q  # get_household_id uses .limit(1).execute()
    user_q.single.return_value = user_q
    # get_household_id expects r.data to be a list of rows: r.data[0].get("household_id")
    user_q.execute.return_value.data = [{"household_id": hh_uuid}]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.gte.return_value = t
        t.order.return_value = t
        t.range.return_value = t
        t.limit.return_value = t
        t.single.return_value = t
        t.execute.return_value.data = []
        t.execute.return_value.count = 0
        if name == "users":
            return user_q
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_get_risk_signal_404_when_not_found(client_risk_signals: TestClient) -> None:
    sig_id = uuid4()
    r = client_risk_signals.get(f"/risk_signals/{sig_id}")
    assert r.status_code in (403, 404)


def test_get_similar_incidents_returns_list(client_risk_signals: TestClient) -> None:
    sig_id = uuid4()
    r = client_risk_signals.get(f"/risk_signals/{sig_id}/similar", params={"top_k": 5})
    assert r.status_code in (200, 403, 404)
    if r.status_code == 200:
        data = r.json()
        assert "similar" in data
        assert isinstance(data["similar"], list)
        assert "available" in data
        assert isinstance(data["available"], bool)
        # When no embedding (mock returns empty), must not compute on synthetic embeddings
        if not data.get("available"):
            assert data.get("reason") == "model_not_run"
            assert data["similar"] == []


def test_submit_feedback_404_when_signal_not_found(client_risk_signals: TestClient) -> None:
    sig_id = uuid4()
    r = client_risk_signals.post(
        f"/risk_signals/{sig_id}/feedback",
        json={"label": "true_positive", "notes": None},
    )
    assert r.status_code in (403, 404)


def test_list_risk_signals_with_query_params(client_risk_signals: TestClient) -> None:
    """GET /risk_signals with status, limit, offset returns 200 and signals/total."""
    r = client_risk_signals.get("/risk_signals", params={"status": "open", "limit": 10, "offset": 0})
    assert r.status_code == 200
    data = r.json()
    assert "signals" in data
    assert "total" in data
    assert isinstance(data["signals"], list)
    assert isinstance(data["total"], int)
    assert data["total"] >= 0


def test_list_risk_signals_severity_min_param(client_risk_signals: TestClient) -> None:
    """GET /risk_signals with severity>= (alias) accepts 1-5 and returns 200."""
    r = client_risk_signals.get("/risk_signals", params={"severity>=": 3})
    assert r.status_code == 200
    data = r.json()
    assert "signals" in data
    assert "total" in data
