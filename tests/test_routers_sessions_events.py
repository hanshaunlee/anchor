"""Tests for sessions router: list_session_events (mocked)."""
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user


@pytest.fixture
def client_sessions_events():
    hh_uuid = str(uuid4())
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.single.return_value = user_q
    user_q.execute.return_value.data = {"household_id": hh_uuid}

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.range.return_value = t
        t.single.return_value = t
        t.execute.return_value.data = []
        t.execute.return_value.count = 0
        if name == "users":
            return user_q
        if name == "sessions":
            t.execute.return_value.data = None
            return t
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_list_session_events_returns_empty_when_no_session(client_sessions_events: TestClient) -> None:
    session_id = uuid4()
    r = client_sessions_events.get(f"/sessions/{session_id}/events")
    assert r.status_code == 200
    data = r.json()
    assert "events" in data
    assert "total" in data
    assert data["total"] == 0
    assert data["events"] == []
