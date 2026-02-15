"""Tests for api.routers.agents: run_financial_agent, get_agents_status, get_financial_trace (mocked)."""
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user


@pytest.fixture
def client_agents():
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.limit.return_value = user_q
    user_q.execute.return_value.data = [{"household_id": "hh-1"}]
    sess_q = MagicMock()
    sess_q.select.return_value = sess_q
    sess_q.eq.return_value = sess_q
    sess_q.order.return_value = sess_q
    sess_q.limit.return_value = sess_q
    sess_q.execute.return_value.data = [{"consent_state": {"share_with_caregiver": True}}]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.execute.return_value.data = [] if name != "sessions" else [{"consent_state": {}}]
        if name == "users":
            return user_q
        if name == "sessions":
            return sess_q
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_financial_agent_run_dry_run(client_agents: TestClient) -> None:
    r = client_agents.post("/agents/financial/run", json={"dry_run": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("dry_run") is True
    assert "risk_signals_count" in data
    assert "watchlists_count" in data


def test_agents_status(client_agents: TestClient) -> None:
    r = client_agents.get("/agents/status")
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data
    assert isinstance(data["agents"], list)


def test_agents_status_includes_last_run_id(client_agents: TestClient) -> None:
    """GET /agents/status returns last_run_id for each agent when present."""
    run_id = str(uuid4())
    agent_runs_row = {
        "id": run_id,
        "agent_name": "supervisor",
        "started_at": "2024-01-15T10:00:00Z",
        "ended_at": "2024-01-15T10:01:00Z",
        "status": "completed",
        "summary_json": {"counts": {"new_signals": 1}},
    }
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.limit.return_value = user_q
    user_q.execute.return_value.data = [{"household_id": "hh-1"}]
    agent_runs_q = MagicMock()
    agent_runs_q.select.return_value = agent_runs_q
    agent_runs_q.eq.return_value = agent_runs_q
    agent_runs_q.order.return_value = agent_runs_q
    agent_runs_q.execute.return_value.data = [agent_runs_row]

    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.execute.return_value.data = []
        if name == "users":
            return user_q
        if name == "agent_runs":
            return agent_runs_q
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        r = client_agents.get("/agents/status")
        assert r.status_code == 200
        data = r.json()
        supervisor = next((a for a in data["agents"] if a.get("agent_name") == "supervisor"), None)
        assert supervisor is not None
        assert supervisor.get("last_run_id") == run_id
        assert supervisor.get("last_run_status") == "completed"
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client_no_auth():
    """Plain client for routes that do not require auth (e.g. GET /agents/financial/demo)."""
    with TestClient(app) as c:
        yield c


def test_financial_demo_no_auth(client_no_auth: TestClient) -> None:
    """GET /agents/financial/demo does not require auth; returns demo run with input_events and output."""
    r = client_no_auth.get("/agents/financial/demo")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert "input_events" in data
    assert len(data["input_events"]) == 3
    assert "Medicare" in data.get("input_summary", "")
    assert "output" in data
    assert "risk_signals" in data["output"]
    assert "watchlists" in data["output"]
    assert "motif_tags" in data["output"]
    assert data.get("risk_signals_count") == len(data["output"]["risk_signals"])
    assert data.get("watchlists_count") == len(data["output"]["watchlists"])


def test_financial_agent_run_use_demo_events(client_agents: TestClient) -> None:
    """POST /agents/financial/run with use_demo_events=true returns input_events and full risk_signals/watchlists."""
    r = client_agents.post("/agents/financial/run", json={"dry_run": True, "use_demo_events": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("use_demo_events") is True
    assert "input_events" in data
    assert len(data["input_events"]) == 3
    assert data.get("risk_signals") is not None
    assert data.get("watchlists") is not None
    assert data.get("risk_signals_count") >= 1


def test_financial_trace(client_agents: TestClient) -> None:
    """GET /agents/financial/trace?run_id=... returns run details when found."""
    run_id = str(uuid4())
    trace_row = {
        "id": run_id,
        "household_id": "hh-1",
        "agent_name": "financial_security",
        "started_at": "2024-01-15T10:00:00Z",
        "ended_at": "2024-01-15T10:01:00Z",
        "status": "completed",
        "summary_json": {"risk_signals_count": 2},
        "step_trace": [
            {"step": "ingest", "status": "ok"},
            {"step": "normalize", "status": "ok"},
            {"step": "detect", "status": "ok"},
            {"step": "recommend_watchlist", "status": "ok"},
            {"step": "persist", "status": "ok"},
        ],
    }
    # Override agent_runs query to return our row; get_household_id uses .limit(1) and r.data[0]
    def table(name):
        from unittest.mock import MagicMock
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.single.return_value = t
        t.limit.return_value = t
        if name == "agent_runs":
            t.execute.return_value.data = trace_row
        else:
            t.execute.return_value.data = [] if name != "sessions" else [{"consent_state": {}}]
        if name == "users":
            t.execute.return_value.data = [{"household_id": "hh-1"}]
        if name == "sessions":
            t.execute.return_value.data = [{"consent_state": {}}]
        return t
    client_agents.app.dependency_overrides.clear()
    mock_sb = MagicMock()
    mock_sb.table.side_effect = table
    from api.deps import get_supabase, require_user
    client_agents.app.dependency_overrides[get_supabase] = lambda: mock_sb
    client_agents.app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        r = client_agents.get(f"/agents/financial/trace?run_id={run_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == run_id
        assert data["agent_name"] == "financial_security"
        assert data["status"] == "completed"
        assert data["summary_json"]["risk_signals_count"] == 2
        assert "step_trace" in data
        assert len(data["step_trace"]) == 5
        assert data["step_trace"][0]["step"] == "ingest"
    finally:
        client_agents.app.dependency_overrides.clear()
