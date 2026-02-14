"""Tests for ingest router: POST /ingest/events (mocked)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user


@pytest.fixture
def client_ingest():
    hh_uuid = str(uuid4())
    session_uuid = uuid4()
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.limit.return_value = user_q
    user_q.execute.return_value.data = [{"household_id": hh_uuid}]

    sessions_q = MagicMock()
    sessions_q.select.return_value = sessions_q
    sessions_q.in_.return_value = sessions_q
    sessions_q.execute.return_value.data = [{"id": str(session_uuid), "household_id": hh_uuid}]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.in_.return_value = t
        t.execute.return_value.data = []
        if name == "users":
            return user_q
        if name == "sessions":
            return sessions_q
        if name == "events":
            t.insert.return_value = t
            t.execute.return_value.data = [{"id": str(uuid4())}]
            return t
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            yield c, session_uuid
    finally:
        app.dependency_overrides.clear()


def test_ingest_events_empty_body(client_ingest: tuple) -> None:
    client, _ = client_ingest
    r = client.post("/ingest/events", json={"events": []})
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] == 0
    assert data["session_ids"] == []
    assert data["last_ts"] is None


def test_ingest_events_with_one_event(client_ingest: tuple) -> None:
    client, session_uuid = client_ingest
    did = uuid4()
    ts = datetime.now(timezone.utc).isoformat()
    body = {
        "events": [
            {
                "session_id": str(session_uuid),
                "device_id": str(did),
                "ts": ts,
                "seq": 0,
                "event_type": "wake",
                "payload": {},
            }
        ]
    }
    r = client.post("/ingest/events", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["ingested"] == 1
    assert str(session_uuid) in data["session_ids"]


def test_ingest_events_403_when_session_not_in_household() -> None:
    """When a session belongs to another household, ingest returns 403."""
    hh_user = str(uuid4())
    other_hh = str(uuid4())
    session_other = uuid4()
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.single.return_value = user_q
    user_q.execute.return_value.data = {"household_id": hh_user}
    sessions_q = MagicMock()
    sessions_q.select.return_value = sessions_q
    sessions_q.in_.return_value = sessions_q
    sessions_q.execute.return_value.data = [{"id": str(session_other), "household_id": other_hh}]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.in_.return_value = t
        t.execute.return_value.data = []
        if name == "users":
            return user_q
        if name == "sessions":
            return sessions_q
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as client:
            r = client.post(
                "/ingest/events",
                json={
                    "events": [
                        {
                            "session_id": str(session_other),
                            "device_id": str(uuid4()),
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "seq": 0,
                            "event_type": "wake",
                            "payload": {},
                        }
                    ]
                },
            )
        assert r.status_code == 403
        assert "household" in r.json().get("detail", "").lower()
    finally:
        app.dependency_overrides.clear()
