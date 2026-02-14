"""
Load ML training config from YAML. Single source for model and graph schema; paths resolved from repo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Repo root (parent of ml/)
_ML_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ML_ROOT.parent


def _resolve_config_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        # Try cwd first, then repo root, then relative to ml/
        for base in [Path.cwd(), _REPO_ROOT, _ML_ROOT]:
            candidate = base / path
            if candidate.exists():
                return candidate
        return Path.cwd() / path
    return p


def load_ml_yaml(config_path: str) -> dict[str, Any]:
    """Load YAML config; resolve path; return dict. Returns {} if file missing or invalid."""
    path = _resolve_config_path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Normalize edge_types: list of lists -> list of tuples
    graph = data.get("graph") or {}
    edge_types = graph.get("edge_types") or []
    if edge_types and isinstance(edge_types[0], list):
        graph["edge_types"] = [tuple(e) for e in edge_types]
    data["graph"] = graph
    return data


def get_train_config(config_path: str | None = None) -> dict[str, Any]:
    """Full config for training; env ANCHOR_ML_CONFIG overrides path."""
    path = config_path or os.environ.get("ANCHOR_ML_CONFIG", "configs/hgt_baseline.yaml")
    # Also try ml/configs/ for default
    if path == "configs/hgt_baseline.yaml" and not _resolve_config_path(path).exists():
        path = "ml/configs/hgt_baseline.yaml"
    return load_ml_yaml(path)
