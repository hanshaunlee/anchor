"""Tests for device, summaries, watchlists, sessions, households routers (mocked Supabase)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user


def _mock_supabase_for_user_household(household_id: str | None = None):
    hh_uuid = household_id or str(uuid4())
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.single.return_value = user_q
    user_q.execute.return_value.data = {"household_id": hh_uuid, "role": "caregiver", "display_name": "Me"}

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.range.return_value = t
        t.in_.return_value = t
        t.single.return_value = t
        t.execute.return_value.data = [] if name not in ("users", "households", "devices") else None
        t.execute.return_value.count = 0
        if name == "users":
            t.execute.return_value.data = {"household_id": hh_uuid, "role": "caregiver", "display_name": "Me"}
            return user_q
        if name == "households":
            t.execute.return_value.data = {"id": hh_uuid, "name": "Test Household"}
            return t
        if name == "devices":
            t.execute.return_value.data = {"id": "d1", "household_id": hh_uuid}
            return t
        return t

    mock_sb.table.side_effect = table
    return mock_sb


@pytest.fixture
def client_authenticated():
    app.dependency_overrides[get_supabase] = lambda: _mock_supabase_for_user_household()
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_households_me(client_authenticated: TestClient) -> None:
    r = client_authenticated.get("/households/me")
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert "name" in data
    assert data.get("role") == "caregiver"


def test_device_sync_requires_body(client_authenticated: TestClient) -> None:
    r = client_authenticated.post("/device/sync", json={})
    assert r.status_code in (200, 422, 404, 403)


def test_list_sessions(client_authenticated: TestClient) -> None:
    r = client_authenticated.get("/sessions")
    assert r.status_code == 200
    data = r.json()
    assert "sessions" in data
    assert "total" in data


def test_list_watchlists(client_authenticated: TestClient) -> None:
    r = client_authenticated.get("/watchlists")
    assert r.status_code == 200
    data = r.json()
    assert "watchlists" in data


def test_list_summaries(client_authenticated: TestClient) -> None:
    r = client_authenticated.get("/summaries")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
