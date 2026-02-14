"""Tests for device router: POST /device/sync (success, 404 device not found, 403 device not in household)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

# DeviceSyncRequest.device_id must be a UUID
TEST_DEVICE_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

pytest.importorskip("supabase")
from fastapi.testclient import TestClient

from api.main import app
from api.deps import get_supabase, require_user


def _mock_sb_device_sync(device_found: bool, same_household: bool, watchlists_data: list | None = None):
    hh_user = str(uuid4())
    hh_device = hh_user if same_household else str(uuid4())
    device_row = {"id": TEST_DEVICE_UUID, "household_id": hh_device} if device_found else None
    user_row = {"household_id": hh_user}
    state_row = {
        "last_upload_ts": "2024-01-15T10:00:00Z",
        "last_upload_seq_by_session": {},
        "last_watchlist_pull_at": "2024-01-15T10:00:00Z",
    }
    wl_data = watchlists_data or []

    def make_table(data):
        t = MagicMock()
        t.select.return_value = t
        t.eq.return_value = t
        t.order.return_value = t
        t.single.return_value = t
        t.upsert.return_value = t
        t.execute.return_value.data = data
        t.execute.return_value.count = 0
        return t

    def table(name):
        if name == "devices":
            return make_table(device_row)
        if name == "users":
            return make_table(user_row)
        if name == "device_sync_state":
            return make_table(state_row)
        if name == "watchlists":
            return make_table(wl_data)
        return make_table(None)

    mock = MagicMock()
    mock.table.side_effect = table
    return mock


@pytest.fixture
def client_device():
    app.dependency_overrides[require_user] = lambda: "user-123"
    try:
        yield app
    finally:
        app.dependency_overrides.clear()


def test_device_sync_success(client_device) -> None:
    """POST /device/sync with valid device in same household returns 200 and DeviceSyncResponse shape."""
    mock_sb = _mock_sb_device_sync(device_found=True, same_household=True, watchlists_data=[])
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    with TestClient(app) as client:
        r = client.post(
            "/device/sync",
            json={
                "device_id": TEST_DEVICE_UUID,
                "last_upload_ts": None,
                "last_upload_seq_by_session": {},
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert "watchlists_delta" in data
    assert "last_upload_ts" in data
    assert "last_upload_seq_by_session" in data
    assert "last_watchlist_pull_at" in data
    assert isinstance(data["watchlists_delta"], list)


def test_device_sync_404_when_device_not_found(client_device) -> None:
    """POST /device/sync when device id not in DB returns 404."""
    mock_sb = _mock_sb_device_sync(device_found=False, same_household=True)
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    with TestClient(app) as client:
        r = client.post(
            "/device/sync",
            json={
                "device_id": str(uuid4()),
                "last_upload_ts": None,
                "last_upload_seq_by_session": {},
            },
        )
    assert r.status_code == 404
    assert "not found" in r.json().get("detail", "").lower() or "detail" in r.json()


def test_device_sync_403_when_device_different_household(client_device) -> None:
    """POST /device/sync when device belongs to another household returns 403."""
    mock_sb = _mock_sb_device_sync(device_found=True, same_household=False)
    app.dependency_overrides[get_supabase] = lambda: mock_sb
    with TestClient(app) as client:
        r = client.post(
            "/device/sync",
            json={
                "device_id": TEST_DEVICE_UUID,
                "last_upload_ts": None,
                "last_upload_seq_by_session": {},
            },
        )
    assert r.status_code == 403
    assert "household" in r.json().get("detail", "").lower()
