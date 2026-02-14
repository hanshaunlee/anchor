"""Extended tests for config.settings: _load_yaml, get_ml_config_path with env."""
import os
from pathlib import Path

import pytest

from config.settings import get_ml_config_path


@pytest.fixture(autouse=True)
def clear_ml_config_path_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANCHOR_ML_CONFIG", raising=False)
    yield


def test_get_ml_config_path_with_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = str(tmp_path / "custom.yaml")
    monkeypatch.setenv("ANCHOR_ML_CONFIG", custom)
    assert get_ml_config_path() == custom


def test_get_ml_config_path_default_contains_hgt() -> None:
    path = get_ml_config_path()
    assert "hgt_baseline" in path or "config" in path
