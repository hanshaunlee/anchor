"""Tests: playbooks, capabilities API, device sync high_risk_mode, connector 501."""
from unittest.mock import MagicMock

import pytest

from fastapi.testclient import TestClient

pytest.importorskip("supabase")
from api.main import app
from api.deps import get_supabase, require_user

HH_ID = "00000000-0000-0000-0000-000000000001"
DEVICE_ID = "00000000-0000-0000-0000-000000000002"
WATCHLIST_ID = "00000000-0000-0000-0000-000000000003"


@pytest.fixture
def client_authenticated():
    mock_sb = MagicMock()
    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.limit.return_value = t
        t.single.return_value = t
        t.range.return_value = t
        t.execute.return_value.data = []
        t.execute.return_value.count = 0
        if name == "users":
            t.execute.return_value.data = [{"household_id": HH_ID}]
        if name == "household_capabilities":
            t.execute.return_value.data = [{
                "household_id": HH_ID,
                "notify_sms_enabled": False,
                "notify_email_enabled": False,
                "device_policy_push_enabled": True,
                "bank_data_connector": "none",
                "bank_control_capabilities": {"lock_card": False, "enable_alerts": True},
                "updated_at": "2024-01-01T00:00:00Z",
            }]
        if name == "action_playbooks":
            t.execute.return_value.data = []
        t.upsert.return_value.execute.return_value.data = []
        t.insert.return_value.execute.return_value.data = [{"id": "id1"}]
        return t
    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_capabilities_me(client_authenticated: TestClient) -> None:
    r = client_authenticated.get("/capabilities/me")
    assert r.status_code == 200
    data = r.json()
    assert "household_id" in data
    assert "bank_control_capabilities" in data
    assert "lock_card" in data.get("bank_control_capabilities", {})


def test_connectors_plaid_link_token_501_without_config() -> None:
    """Without PLAID_CLIENT_ID/SECRET, Plaid link_token returns 501."""
    import os
    had_id = os.environ.pop("PLAID_CLIENT_ID", None)
    had_secret = os.environ.pop("PLAID_SECRET", None)
    try:
        app.dependency_overrides[require_user] = lambda: "user-123"
        mock_sb = MagicMock()
        mock_sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [{"household_id": "hh1"}]
        app.dependency_overrides[get_supabase] = lambda: mock_sb
        with TestClient(app) as c:
            r = c.get("/connectors/plaid/link_token", headers={"Authorization": "Bearer fake"})
        assert r.status_code == 501
        assert "not configured" in r.json().get("detail", "").lower() or "plaid" in r.json().get("detail", "").lower()
    finally:
        if had_id is not None:
            os.environ["PLAID_CLIENT_ID"] = had_id
        if had_secret is not None:
            os.environ["PLAID_SECRET"] = had_secret
        app.dependency_overrides.clear()


def test_device_sync_returns_high_risk_mode_when_watchlist_has_it() -> None:
    """Device sync response includes high_risk_mode when watchlist has watch_type=high_risk_mode and active."""
    from datetime import datetime, timezone
    mock_sb = MagicMock()
    def table(name):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.upsert.return_value.execute.return_value.data = []
        t.single.return_value = t
        t.execute.return_value.data = []
        if name == "users":
            t.execute.return_value.data = {"household_id": HH_ID}
        if name == "devices":
            t.execute.return_value.data = {"id": DEVICE_ID, "household_id": HH_ID}
        if name == "device_sync_state":
            t.execute.return_value.data = {
                "last_upload_ts": None,
                "last_upload_seq_by_session": {},
                "last_watchlist_pull_at": datetime.now(timezone.utc).isoformat(),
            }
        if name == "watchlists":
            t.execute.return_value.data = [
                {
                    "id": WATCHLIST_ID,
                    "watch_type": "high_risk_mode",
                    "pattern": {"active": True, "reason": "incident_response", "risk_signal_id": "rs1"},
                    "reason": "High-risk mode active.",
                    "priority": 100,
                    "expires_at": None,
                }
            ]
        return t
    mock_sb.table.side_effect = table
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        with TestClient(app) as c:
            r = c.post(
                "/device/sync",
                json={
                    "device_id": DEVICE_ID,
                    "last_upload_ts": None,
                    "last_upload_seq_by_session": {},
                },
                headers={"Authorization": "Bearer fake"},
            )
        assert r.status_code == 200, (r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
        data = r.json()
        assert "high_risk_mode" in data
        assert data["high_risk_mode"] is not None
        assert data["high_risk_mode"].get("active") is True
    finally:
        app.dependency_overrides.clear()
