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
    consent_allow_outbound_key: str = Field("consent_allow_outbound_contact", description="Consent key for outbound caregiver contact (SMS/email/voice)")
    default_consent_share: bool = Field(True, description="Default when consent not set")
    default_consent_watchlist: bool = Field(True, description="Default when consent not set")
    default_consent_allow_outbound: bool = Field(False, description="Default for consent_allow_outbound_contact (opt-in for action-taking)")
    # caregiver_contact_policy (JSONB): allowed_channels, quiet_hours, escalation_threshold; defaults in code when empty
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


class AgentSettings(BaseSettings):
    """Domain agents and services: limits, windows, thresholds (env: ANCHOR_AGENT_*)."""

    model_config = SettingsConfigDict(env_prefix="ANCHOR_AGENT_", env_file=".env", extra="ignore")

    # Evidence narrative
    evidence_signal_limit: int = Field(20, ge=1, le=500, description="Max open signals to process per run")
    evidence_llm_max_tokens: int = Field(400, ge=1, le=2000, description="LLM narrative max tokens")

    # Graph drift
    drift_window_recent_days: int = Field(3, ge=1, le=90)
    drift_window_baseline_days: int = Field(14, ge=1, le=365)
    drift_baseline_end_days: int = Field(7, ge=0, le=90)
    drift_min_samples_per_window: int = Field(10, ge=2, le=1000)
    drift_threshold: float = Field(0.15, ge=0.0, le=1.0)
    drift_top_prototype_examples: int = Field(3, ge=1, le=20)
    drift_k_neighbors: int = Field(5, ge=1, le=50)
    drift_pca_components: int = Field(10, ge=1, le=64)
    drift_mmd_threshold: float = Field(0.2, ge=0.0, le=1.0)
    drift_ks_threshold: float = Field(0.3, ge=0.0, le=1.0)
    drift_severity_high_centroid: float = Field(0.25, ge=0.0, le=1.0)

    # Continual calibration
    calibration_min_labeled: int = Field(10, ge=5, le=1000)
    calibration_target_fpr: float = Field(0.1, ge=0.01, le=0.5)
    calibration_ece_bins: int = Field(10, ge=2, le=20)

    # Ring discovery
    ring_min_community_size: int = Field(2, ge=2, le=20)
    ring_top_rings: int = Field(10, ge=1, le=50)
    ring_novelty_days: int = Field(7, ge=1, le=90)
    ring_lookback_days: int = Field(30, ge=7, le=365)

    # Similarity service
    similar_incidents_window_days: int = Field(90, ge=1, le=365)
    similar_incidents_top_k: int = Field(5, ge=1, le=20)

    # Risk scoring (explainer subgraph)
    risk_scoring_top_k_edges: int = Field(20, ge=1, le=100)


class NotifySettings(BaseSettings):
    """Outbound notify provider (env: ANCHOR_NOTIFY_*)."""

    model_config = SettingsConfigDict(env_prefix="ANCHOR_NOTIFY_", env_file=".env", extra="ignore")

    provider: str = Field("mock", description="mock | twilio | sendgrid | smtp")
    # Twilio: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM (set in env)
    # SendGrid: SENDGRID_API_KEY, SENDGRID_FROM
    # SMTP: SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_FROM


class WorkerSettings(BaseSettings):
    """Worker job flags (env: ANCHOR_WORKER_*)."""

    model_config = SettingsConfigDict(env_prefix="ANCHOR_WORKER_", env_file=".env", extra="ignore")

    outreach_auto_trigger: bool = Field(True, description="After persisting risk_signals, run outreach for severity >= threshold when consent allows")


@lru_cache
def get_notify_settings() -> NotifySettings:
    return NotifySettings()


@lru_cache
def get_worker_settings() -> WorkerSettings:
    return WorkerSettings()


@lru_cache
def get_pipeline_settings() -> PipelineSettings:
    return PipelineSettings()


@lru_cache
def get_ml_settings() -> MLSettings:
    return MLSettings()


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_ml_config_path() -> str:
    """Config path for HGT/training; respects env override."""
    return os.environ.get("ANCHOR_ML_CONFIG", "configs/hgt_baseline.yaml")
