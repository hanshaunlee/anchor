"""
Model registry: single interface for risk scoring and retrieval embeddings.
Select model via ANCHOR_GNN_MODEL (e.g. hgt | hgt_baseline | fraudgt_entity).
Enables a two-model stack later: HGT for risk + FraudGT-style for entity retrieval.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ModelRunner(Protocol):
    """Interface all GNN risk models implement for the scoring pipeline."""

    def run(
        self,
        household_id: str,
        *,
        sessions: list[dict],
        utterances: list[dict],
        entities: list[dict],
        mentions: list[dict],
        relationships: list[dict],
        devices: list[dict] | None = None,
        events: list[dict] | None = None,
    ) -> tuple[list[dict], str | None]:
        """
        Build graph from tables, run forward, return (raw_scores, target_node_type).
        raw_scores: list of dicts with node_type, node_index, score, label, embedding (optional).
        target_node_type: e.g. "entity".
        """
        ...


def _default_checkpoint_path() -> Path:
    try:
        from config.settings import get_ml_settings
        return Path(get_ml_settings().checkpoint_path)
    except Exception:
        return Path("runs/hgt_baseline/best.pt")


class _HGTRunner:
    """HGT baseline: heterogeneous graph, risk scores + retrieval embeddings."""

    def __init__(self, checkpoint_path: Path | None = None):
        self._path = checkpoint_path or _default_checkpoint_path()
        self._model = None
        self._target_node_type = None
        self._device = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self._path.is_file():
            logger.debug("HGT checkpoint missing at %s", self._path)
            return False
        try:
            import torch
            from ml.inference import load_model
            device = torch.device("cpu")
            self._model, self._target_node_type = load_model(self._path, device)
            self._device = device
            return True
        except Exception as e:
            logger.debug("HGT load failed: %s", e)
            return False

    def run(
        self,
        household_id: str,
        *,
        sessions: list[dict],
        utterances: list[dict],
        entities: list[dict],
        mentions: list[dict],
        relationships: list[dict],
        devices: list[dict] | None = None,
        events: list[dict] | None = None,
    ) -> tuple[list[dict], str | None]:
        if not entities:
            return [], None
        if not self._ensure_loaded():
            return [], None
        try:
            import torch
            from ml.inference import run_inference
            from ml.graph.builder import build_hetero_from_tables
            devices = devices or []
            data = build_hetero_from_tables(
                household_id,
                sessions,
                utterances,
                entities,
                mentions,
                relationships,
                devices=devices,
            )
            raw_scores, _ = run_inference(
                self._model,
                data,
                self._device,
                target_node_type=self._target_node_type,
                return_embeddings=True,
            )
            return raw_scores, self._target_node_type or "entity"
        except Exception as e:
            logger.debug("HGT inference failed: %s", e)
            return [], None


class _FraudGTRunner:
    """FraudGT-style entity graph: edge-aware, for rings/similarity. Not wired to checkpoint yet."""

    def run(
        self,
        household_id: str,
        *,
        sessions: list[dict],
        utterances: list[dict],
        entities: list[dict],
        mentions: list[dict],
        relationships: list[dict],
        devices: list[dict] | None = None,
        events: list[dict] | None = None,
    ) -> tuple[list[dict], str | None]:
        # TODO: build entity-only graph, load FraudGT checkpoint, return embeddings (no risk logits)
        logger.debug("FraudGT entity runner not implemented for pipeline; use HGT")
        return [], None


def get_model_name() -> str:
    """Model name from env ANCHOR_GNN_MODEL; default hgt."""
    try:
        import os
        name = os.environ.get("ANCHOR_GNN_MODEL", "").strip().lower()
        return name or "hgt"
    except Exception:
        return "hgt"


def get_runner(model_name: str | None = None, checkpoint_path: Any = None) -> ModelRunner | None:
    """
    Return a ModelRunner for the given model name.
    model_name: hgt | hgt_baseline -> HGT risk + embeddings; fraudgt_entity -> stub (not implemented).
    """
    name = (model_name or get_model_name()).lower()
    path = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path) if not isinstance(checkpoint_path, Path) else checkpoint_path
    if name in ("hgt", "hgt_baseline", ""):
        return _HGTRunner(checkpoint_path=path)
    if name in ("fraudgt_entity", "fraudgt"):
        return _FraudGTRunner()
    logger.warning("Unknown ANCHOR_GNN_MODEL=%s; falling back to HGT", model_name)
    return _HGTRunner(checkpoint_path=path)
