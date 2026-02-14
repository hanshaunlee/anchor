"""Tests for api.config: Settings."""
import pytest

from api.config import settings


def test_settings_has_expected_attrs() -> None:
    assert hasattr(settings, "supabase_url")
    assert hasattr(settings, "supabase_service_role_key")
    assert hasattr(settings, "database_url")
    assert hasattr(settings, "jwt_secret")


def test_settings_defaults_empty_strings() -> None:
    assert isinstance(settings.supabase_url, str)
    assert isinstance(settings.supabase_service_role_key, str)
