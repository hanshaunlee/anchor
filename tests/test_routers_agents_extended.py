"""Extended tests for agents router: drift, narrative, ring, calibration, redteam endpoints and GET /agents/trace."""
from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user


@pytest.fixture
def client_agents():
    """Same pattern as test_routers_agents.client_agents so get_household_id returns hh-1."""
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
    sess_q.execute.return_value.data = [{"consent_state": {}}]

    def table(name):
        if name == "users":
            return user_q
        if name == "sessions":
            return sess_q
        if name == "agent_runs":
            agent_t = MagicMock()
            agent_t.select.return_value = agent_t
            agent_t.eq.return_value = agent_t
            agent_t.single.return_value = agent_t
            agent_t.execute.return_value.data = None
            return agent_t
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.range.return_value = t
        t.insert.return_value = t
        t.update.return_value.eq.return_value.execute.return_value = None
        t.execute.return_value.data = [] if name != "sessions" else [{"consent_state": {}}]
        t.execute.return_value.count = 0
        return t

    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_post_drift_run_returns_ok_and_step_trace(client_agents: TestClient) -> None:
    """POST /agents/drift/run returns 200, ok=True, step_trace and summary_json."""
    r = client_agents.post("/agents/drift/run", json={"dry_run": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("dry_run") is True
    assert "step_trace" in data
    assert "summary_json" in data
    assert isinstance(data["step_trace"], list)
    assert "fetch_embeddings" in [s.get("step") for s in data["step_trace"]]


def test_post_narrative_run_returns_ok(client_agents: TestClient) -> None:
    """POST /agents/narrative/run returns 200, ok, step_trace, summary_json."""
    r = client_agents.post("/agents/narrative/run", json={"dry_run": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert "step_trace" in data
    assert "summary_json" in data


def test_post_ring_run_returns_ok(client_agents: TestClient) -> None:
    """POST /agents/ring/run returns 200, step_trace includes check_neo4j/gds."""
    r = client_agents.post("/agents/ring/run", json={"dry_run": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    steps = [s.get("step") for s in data.get("step_trace", [])]
    assert "check_neo4j" in steps or "gds_similarity" in steps


def test_post_calibration_run_returns_ok(client_agents: TestClient) -> None:
    """POST /agents/calibration/run returns 200, summary_json has report shape."""
    r = client_agents.post("/agents/calibration/run", json={"dry_run": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert "summary_json" in data


def test_post_redteam_run_returns_ok(client_agents: TestClient) -> None:
    """POST /agents/redteam/run returns 200, summary has variants_generated/regression_passed."""
    r = client_agents.post("/agents/redteam/run", json={"dry_run": True})
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    assert data.get("summary_json", {}).get("regression_passed") is True


def test_agents_drift_run_403_when_no_household() -> None:
    """POST /agents/drift/run returns 403 when user has no household."""
    mock_sb = MagicMock()
    user_q = MagicMock()
    user_q.select.return_value = user_q
    user_q.eq.return_value = user_q
    user_q.single.return_value = user_q
    user_q.execute.return_value.data = None
    mock_sb.table.side_effect = lambda name: user_q if name == "users" else MagicMock()
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as client:
            r = client.post("/agents/drift/run", json={})
        assert r.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_agents_trace_generic_requires_run_id_and_agent_name(client_agents: TestClient) -> None:
    """GET /agents/trace requires run_id and agent_name query params."""
    r = client_agents.get("/agents/trace")
    assert r.status_code == 422
    r = client_agents.get("/agents/trace?run_id=00000000-0000-0000-0000-000000000000&agent_name=graph_drift")
    assert r.status_code in (200, 404)


def test_agents_trace_404_when_run_not_found(client_agents: TestClient) -> None:
    """GET /agents/trace returns 404 when run_id not in DB for household (agent_runs returns no row)."""
    run_id = uuid4()
    r = client_agents.get(f"/agents/trace?run_id={run_id}&agent_name=graph_drift")
    assert r.status_code == 404


def test_agents_status_includes_all_known_agents(client_agents: TestClient) -> None:
    """GET /agents/status returns list of KNOWN_AGENTS (financial_security, graph_drift, etc.) when household exists."""
    r = client_agents.get("/agents/status")
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data
    agents = data["agents"]
    names = [a["agent_name"] for a in agents]
    assert "financial_security" in names
    assert "graph_drift" in names
    assert "evidence_narrative" in names
    assert "ring_discovery" in names
    assert "continual_calibration" in names
    assert "synthetic_redteam" in names
