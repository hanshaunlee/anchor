"""Tests for API routers: health, risk_signals (with mocked auth/Supabase)."""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("supabase")
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import app after path setup in conftest
from api.main import app
from api.deps import get_supabase, require_user


def _override_get_supabase_503():
    """Match real app: Supabase not configured -> 503."""
    raise HTTPException(status_code=503, detail="Supabase not configured")


def _override_require_user():
    return "user-123"


@pytest.fixture
def client_with_user():
    mock_supabase = MagicMock()
    q = mock_supabase.table.return_value
    q.select.return_value = q
    q.eq.return_value = q
    q.order.return_value = q
    q.range.return_value = q
    q.execute.return_value.data = []
    q.execute.return_value.count = 0
    # users table for _household_id
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.single.return_value = user_q
    user_q.execute.return_value.data = {"household_id": "hh-1"}
    def table(name):
        if name == "users":
            return user_q
        return q
    mock_supabase.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_supabase
    app.dependency_overrides[require_user] = _override_require_user
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_risk_signals_list_requires_auth() -> None:
    """Without valid auth or when Supabase not configured, get 401 or 503."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_supabase] = _override_get_supabase_503
    with TestClient(app) as client:
        r = client.get("/risk_signals")
    assert r.status_code in (401, 503)


def test_risk_signals_list_with_mock_returns_empty(client_with_user: TestClient) -> None:
    r = client_with_user.get("/risk_signals")
    assert r.status_code == 200
    data = r.json()
    assert "signals" in data
    assert "total" in data
    assert data["total"] == 0
    assert data["signals"] == []


def test_docs_available() -> None:
    with TestClient(app) as client:
        r = client.get("/docs")
    assert r.status_code == 200


def test_redoc_available() -> None:
    with TestClient(app) as client:
        r = client.get("/redoc")
    assert r.status_code == 200


def test_ingest_events_empty_body_requires_auth() -> None:
    """Without valid auth or when Supabase not configured, get 401 or 503."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_supabase] = _override_get_supabase_503
    with TestClient(app) as client:
        r = client.post("/ingest/events", json={"events": []})
    assert r.status_code in (401, 422, 503)
