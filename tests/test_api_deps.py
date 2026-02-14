"""Tests for api.deps: get_supabase (503 when not configured), get_current_user_id, require_user."""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("supabase")
from fastapi import HTTPException

from api.deps import get_supabase, get_current_user_id, require_user


def test_require_user_raises_401_when_none() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_user(None)
    assert exc_info.value.status_code == 401
    assert "authenticated" in (exc_info.value.detail or "").lower()


def test_require_user_returns_user_id() -> None:
    assert require_user("user-123") == "user-123"


def test_get_supabase_raises_503_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import api.deps as deps_module
    monkeypatch.setattr(deps_module.settings, "supabase_url", "")
    monkeypatch.setattr(deps_module.settings, "supabase_service_role_key", "")
    with pytest.raises(HTTPException) as exc_info:
        get_supabase()
    assert exc_info.value.status_code == 503
    assert "not configured" in (exc_info.value.detail or "").lower()


def test_get_current_user_id_returns_none_when_no_credentials() -> None:
    result = get_current_user_id(credentials=None, supabase=MagicMock())
    assert result is None
