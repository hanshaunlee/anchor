"""Tests for ml.config: load_ml_yaml, get_train_config, path resolution."""
import os
from pathlib import Path

import pytest


def test_load_ml_yaml_missing_returns_empty() -> None:
    from ml.config import load_ml_yaml
    out = load_ml_yaml("/nonexistent/path/does/not/exist.yaml")
    assert out == {}


def test_load_ml_yaml_valid_file(tmp_path: Path) -> None:
    from ml.config import load_ml_yaml
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text("""
model:
  name: hgt
  hidden: 32
graph:
  edge_types:
    - [entity, co_occurs, entity]
""")
    out = load_ml_yaml(str(yaml_path))
    assert out.get("model", {}).get("name") == "hgt"
    assert out.get("model", {}).get("hidden") == 32
    graph = out.get("graph", {})
    edge_types = graph.get("edge_types", [])
    assert len(edge_types) >= 1
    if edge_types and isinstance(edge_types[0], (list, tuple)):
        assert tuple(edge_types[0]) == ("entity", "co_occurs", "entity")


def test_load_ml_yaml_normalizes_edge_types_lists_to_tuples(tmp_path: Path) -> None:
    from ml.config import load_ml_yaml
    yaml_path = tmp_path / "edges.yaml"
    yaml_path.write_text("graph:\n  edge_types:\n    - [a, rel, b]\n")
    out = load_ml_yaml(str(yaml_path))
    et = out.get("graph", {}).get("edge_types", [])
    assert et and isinstance(et[0], tuple)
    assert et[0] == ("a", "rel", "b")


def test_get_train_config_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from ml.config import get_train_config
    monkeypatch.delenv("ANCHOR_ML_CONFIG", raising=False)
    cfg = get_train_config()
    # May be {} if no file at default path, or dict with keys
    assert isinstance(cfg, dict)


def test_get_train_config_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ml.config import get_train_config
    custom = tmp_path / "custom_ml.yaml"
    custom.write_text("model:\n  name: custom\n")
    monkeypatch.setenv("ANCHOR_ML_CONFIG", str(custom))
    cfg = get_train_config()
    assert cfg.get("model", {}).get("name") == "custom"


def test_get_train_config_explicit_path(tmp_path: Path) -> None:
    from ml.config import get_train_config
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("model:\n  hidden: 64\n")
    cfg = get_train_config(config_path=str(explicit))
    assert cfg.get("model", {}).get("hidden") == 64
