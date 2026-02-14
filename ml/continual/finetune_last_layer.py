"""
Stub: fine-tune last layer from caregiver feedback (weekly batch).
Confirming "false positive" reduces future alerts for similar patterns via threshold or watchlist edit.
Even if not fully run at hackathon, provides the hook for household-specific calibration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_feedback_batch(supabase_client: Any, household_id: str, since_ts: str | None = None) -> list[dict]:
    """Load feedback rows (risk_signal_id, label, notes) for household since timestamp."""
    q = (
        supabase_client.table("feedback")
        .select("risk_signal_id, label, notes, created_at")
        .eq("household_id", household_id)
    )
    if since_ts:
        q = q.gte("created_at", since_ts)
    r = q.execute()
    return list(r.data or [])


def apply_threshold_adjustment(supabase_client: Any, household_id: str, adjustment: float) -> None:
    """Store per-household threshold adjustment (immediate effect on severity >= threshold)."""
    supabase_client.table("household_calibration").upsert(
        {
            "household_id": household_id,
            "severity_threshold_adjust": adjustment,
            "updated_at": "now()",
        },
        on_conflict="household_id",
    ).execute()


def finetune_last_layer_stub(
    checkpoint_path: Path,
    feedback_batch: list[dict],
    output_path: Path | None = None,
) -> dict:
    """
    Stub: would fine-tune last layer on (embedding, label) from feedback.
    Returns metrics dict for logging. Actual training loop can be added later.
    """
    logger.info("Finetune stub: %d feedback items, checkpoint %s", len(feedback_batch), checkpoint_path)
    # Placeholder: in production, load model, get embeddings for risk_signal_ids,
    # train last layer with BCE/cross-entropy, save.
    return {"feedback_count": len(feedback_batch), "status": "stub", "saved": False}
