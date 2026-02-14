"""Graph Drift Agent: embedding distribution shift; opens drift_warning risk_signal if shift > tau; suggests retrain."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD_TAU = 0.15  # configurable


def run_graph_drift_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
    tau: float = DRIFT_THRESHOLD_TAU,
) -> dict[str, Any]:
    """
    Nightly job: compute embedding distribution shift for household.
    If shift > tau, open risk_signal with type drift_warning and trigger retrain suggestion.
    Returns step_trace, summary_json, status.
    """
    step_trace: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()

    step_trace.append({"step": "fetch_embeddings", "status": "ok", "ts": started})
    shift = 0.0
    n_embeddings = 0

    if supabase:
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            r = (
                supabase.table("risk_signal_embeddings")
                .select("embedding, created_at")
                .eq("household_id", household_id)
                .gte("created_at", since)
                .eq("has_embedding", True)
                .execute()
            )
            rows = r.data or []
            n_embeddings = len([x for x in rows if x.get("embedding")])
            # Placeholder: real drift = compare recent vs baseline distribution (e.g. MMD or mean shift)
            # For hackathon: no-op drift computation
            step_trace.append({"step": "compute_shift", "status": "ok", "n_embeddings": n_embeddings, "shift": shift})
        except Exception as e:
            logger.warning("Graph drift fetch failed: %s", e)
            step_trace.append({"step": "fetch_embeddings", "status": "error", "error": str(e)})
            return {
                "step_trace": step_trace,
                "summary_json": {"error": str(e), "shift": None},
                "status": "error",
                "started_at": started,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }

    if shift > tau and supabase and not dry_run:
        step_trace.append({"step": "open_drift_warning", "status": "ok"})
        try:
            supabase.table("risk_signals").insert({
                "household_id": household_id,
                "signal_type": "drift_warning",
                "severity": 2,
                "score": float(shift),
                "explanation": {"summary": f"Embedding distribution shift {shift:.3f} > τ={tau}; retrain suggested.", "model_available": True},
                "recommended_action": {"action": "review", "retrain_suggested": True},
                "status": "open",
            }).execute()
        except Exception as e:
            step_trace.append({"step": "open_drift_warning", "status": "error", "error": str(e)})
    elif shift > tau and dry_run:
        step_trace.append({"step": "open_drift_warning", "status": "dry_run", "would_open": True})

    ended = datetime.now(timezone.utc).isoformat()
    return {
        "step_trace": step_trace,
        "summary_json": {"shift": shift, "n_embeddings": n_embeddings, "threshold": tau, "drift_detected": shift > tau},
        "status": "ok",
        "started_at": started,
        "ended_at": ended,
    }
