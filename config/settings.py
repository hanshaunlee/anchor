"""
Pipeline and ML settings from environment. Use env vars to override defaults without code changes.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Risk pipeline thresholds and limits (env: ANCHOR_*)."""

    model_config = SettingsConfigDict(env_prefix="ANCHOR_", env_file=".env", extra="ignore")

    risk_score_threshold: float = Field(0.5, description="Score above which we flag for review")
    explanation_score_min: float = Field(0.4, description="Min score to include in explanations")
    watchlist_score_min: float = Field(0.5, description="Min score to add to watchlist")
    escalation_score_min: float = Field(0.6, description="Min score to draft escalation")
    persist_score_min: float = Field(0.3, description="Min score to persist risk signal")
    severity_threshold: int = Field(4, ge=1, le=5, description="Severity >= this triggers HITL review")
    timeline_snippet_max: int = Field(6, ge=1, le=50, description="Max events in timeline snippet")
    consent_share_key: str = Field("share_with_caregiver", description="Consent key for escalation")
    consent_watchlist_key: str = Field("watchlist_ok", description="Consent key for watchlist")
    default_consent_share: bool = Field(True, description="Default when consent not set")
    default_consent_watchlist: bool = Field(True, description="Default when consent not set")
    # Failure containment: no graph mutation if ASR/intent confidence below this (0 = allow all).
    asr_confidence_min_for_graph: float = Field(0.0, ge=0.0, le=1.0, description="Min ASR/intent confidence to mutate graph")


class MLSettings(BaseSettings):
    """ML and worker defaults (env: ANCHOR_ML_*)."""

    model_config = SettingsConfigDict(env_prefix="ANCHOR_ML_", env_file=".env", extra="ignore")

    embedding_dim: int = Field(32, ge=8, le=512, description="Risk signal embedding dimension")
    risk_inference_entity_cap: int = Field(100, ge=1, le=10000, description="Max entities per inference batch")
    calibration_adjust_step: float = Field(0.1, ge=0.01, le=1.0, description="Threshold adjust on false positive")
    calibration_adjust_cap: float = Field(2.0, description="Max severity_threshold_adjust (false_positive)")
    calibration_adjust_floor: float = Field(-0.5, description="Min severity_threshold_adjust (true_positive)")
    calibration_true_positive_step: float = Field(-0.05, description="Step on true_positive (decrease threshold)")
    model_version_tag: str = Field("v0", description="Model version for embeddings table")
    checkpoint_path: str = Field("runs/hgt_baseline/best.pt", description="Default HGT checkpoint path (env: ANCHOR_ML_CHECKPOINT_PATH)")
    explainer_epochs: int = Field(50, ge=1, le=500, description="Epochs for GNN explainer (mask optimization)")


@lru_cache
def get_pipeline_settings() -> PipelineSettings:
    return PipelineSettings()


@lru_cache
def get_ml_settings() -> MLSettings:
    return MLSettings()


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_ml_config_path() -> str:
    """Config path for HGT/training; respects env override."""
    return os.environ.get("ANCHOR_ML_CONFIG", "configs/hgt_baseline.yaml")
