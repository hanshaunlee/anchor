"""Continual Calibration Agent: temperature scaling + conformal; update household calibration from feedback; produce report."""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def run_continual_calibration_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Update household calibration from feedback; optionally recalibrate classifier head weekly;
    produce calibration report.
    """
    step_trace: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()

    step_trace.append({"step": "fetch_feedback", "status": "ok"})
    report = {"household_id": household_id, "feedback_count": 0, "adjustment_applied": None}

    if supabase:
        try:
            r = (
                supabase.table("feedback")
                .select("id, label", count="exact")
                .eq("household_id", household_id)
                .execute()
            )
            report["feedback_count"] = r.count or 0
            cal = (
                supabase.table("household_calibration")
                .select("severity_threshold_adjust")
                .eq("household_id", household_id)
                .single()
                .execute()
            )
            current = (cal.data or {}).get("severity_threshold_adjust", 0)
            report["current_adjustment"] = current
            # Placeholder: temperature scaling / conformal would use feedback labels vs scores
            if not dry_run and report["feedback_count"] > 0:
                # No change for now; could update calibration params from feedback batch
                report["adjustment_applied"] = "none"
            step_trace.append({"step": "calibration", "status": "ok", "report": report})
        except Exception as e:
            logger.warning("Calibration agent failed: %s", e)
            step_trace.append({"step": "fetch_feedback", "status": "error", "error": str(e)})
            return {
                "step_trace": step_trace,
                "summary_json": {"error": str(e)},
                "status": "error",
                "started_at": started,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }

    return {
        "step_trace": step_trace,
        "summary_json": report,
        "status": "ok",
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
