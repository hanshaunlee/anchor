"""
Continual Calibration Agent: Platt scaling / conformal threshold from feedback.
Updates household_calibration (calibration_params, last_calibrated_at); produces evaluation report.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run, step

logger = logging.getLogger(__name__)

MIN_LABELED = 10
TARGET_FPR = 0.1


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))
    except Exception:
        return 0.0


def _fetch_labeled_data(supabase: Any, household_id: str) -> list[tuple[float, int]]:
    """Join feedback with risk_signals; return list of (score, 1 for TP else 0)."""
    out: list[tuple[float, int]] = []
    try:
        fb = (
            supabase.table("feedback")
            .select("risk_signal_id, label")
            .eq("household_id", household_id)
            .in_("label", ["true_positive", "false_positive"])
            .execute()
        )
        for row in (fb.data or []):
            rsid = row.get("risk_signal_id")
            label = row.get("label")
            if not rsid or label not in ("true_positive", "false_positive"):
                continue
            sig = (
                supabase.table("risk_signals")
                .select("score")
                .eq("id", rsid)
                .eq("household_id", household_id)
                .limit(1)
                .execute()
            )
            if sig.data and len(sig.data) > 0 and sig.data[0].get("score") is not None:
                score = float(sig.data[0]["score"])
                y = 1 if label == "true_positive" else 0
                out.append((score, y))
    except Exception as e:
        logger.debug("Calibration fetch labeled failed: %s", e)
    return out


def _platt_fit(scores: list[float], labels: list[int]) -> tuple[float, float] | None:
    """Logistic regression on score -> P(TP). Returns (a, b) for sigmoid(a*score + b)."""
    if len(scores) < 5 or len(scores) != len(labels):
        return None
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        X = np.array(scores).reshape(-1, 1)
        y = np.array(labels)
        lr = LogisticRegression(C=1e6, max_iter=500).fit(X, y)
        b = float(lr.intercept_[0])
        a = float(lr.coef_[0][0])
        return (a, b)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Platt fit failed: %s", e)
    return None


def _conformal_threshold(scores: list[float], labels: list[int], target_fpr: float) -> float | None:
    """Threshold on score such that FPR on false_positives is ~ target_fpr."""
    fp_scores = [s for s, y in zip(scores, labels) if y == 0]
    if not fp_scores:
        return None
    fp_scores_sorted = sorted(fp_scores, reverse=True)
    idx = max(0, int((1 - target_fpr) * len(fp_scores_sorted)) - 1)
    return fp_scores_sorted[idx] if idx < len(fp_scores_sorted) else fp_scores_sorted[-1]


def _ece(probs: list[float], labels: list[int], n_bins: int = 10) -> float:
    """Expected calibration error."""
    if not probs or len(probs) != len(labels) or n_bins < 1:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bi = min(n_bins - 1, int(p * n_bins))
        bins[bi].append((p, y))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = sum(x[0] for x in b) / len(b)
        avg_y = sum(x[1] for x in b) / len(b)
        ece += len(b) * abs(avg_p - avg_y)
    return ece / len(probs) if probs else 0.0


def run_continual_calibration_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Fetch labeled (feedback + risk_signals score); fit Platt scaling and/or conformal threshold;
    update household_calibration; produce before/after evaluation report.
    When insufficient labels, report reason and do not update.
    """
    step_trace: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)

    if not supabase:
        step_trace.append({
            "step": "fetch_feedback",
            "status": "skip",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": "no_supabase",
        })
        summary = {"feedback_count": 0, "adjustment_applied": None, "reason": "no_supabase"}
        ended_at = datetime.now(timezone.utc).isoformat()
        return {"step_trace": step_trace, "summary_json": summary, "status": "ok", "started_at": started_at, "ended_at": ended_at, "run_id": None}

    with step(ctx, step_trace, "fetch_feedback"):
        labeled = _fetch_labeled_data(supabase, household_id)
        scores = [x[0] for x in labeled]
        labels = [x[1] for x in labeled]
        step_trace[-1]["outputs_count"] = len(labeled)
        step_trace[-1]["notes"] = f"{len(labeled)} labeled (TP={sum(labels)}, FP={len(labels)-sum(labels)})"

    report: dict[str, Any] = {
        "household_id": household_id,
        "feedback_count": len(labeled),
        "adjustment_applied": None,
        "current_adjustment": None,
        "calibration_params": None,
        "before_ece": None,
        "after_ece": None,
        "conformal_threshold": None,
    }

    try:
        cal = (
            supabase.table("household_calibration")
            .select("severity_threshold_adjust, calibration_params")
            .eq("household_id", household_id)
            .single()
            .execute()
        )
        if cal.data:
            report["current_adjustment"] = cal.data.get("severity_threshold_adjust")
            report["calibration_params"] = cal.data.get("calibration_params")
    except Exception:
        pass

    if len(labeled) < MIN_LABELED:
        step_trace.append({
            "step": "calibration",
            "status": "ok",
            "started_at": step_trace[-1]["ended_at"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": f"insufficient_labels (min {MIN_LABELED})",
        })
        report["reason"] = "insufficient_labels"
        report["adjustment_applied"] = "none"
        summary = report
        ended_at = datetime.now(timezone.utc).isoformat()
        run_id = persist_agent_run(supabase, household_id, "continual_calibration", started_at=started_at, ended_at=ended_at, status="completed", step_trace=step_trace, summary_json=summary, dry_run=dry_run)
        return {"step_trace": step_trace, "summary_json": summary, "status": "ok", "started_at": started_at, "ended_at": ended_at, "run_id": run_id}

    with step(ctx, step_trace, "calibration"):
        platt = _platt_fit(scores, labels)
        conf_thresh = _conformal_threshold(scores, labels, TARGET_FPR)
        probs_raw = [_sigmoid(s) for s in scores]
        report["before_ece"] = round(_ece(probs_raw, labels), 4)
        calibration_params: dict[str, Any] = {}
        if platt:
            calibration_params["platt_a"] = platt[0]
            calibration_params["platt_b"] = platt[1]
            probs_cal = [_sigmoid(platt[0] * s + platt[1]) for s in scores]
            report["after_ece"] = round(_ece(probs_cal, labels), 4)
        if conf_thresh is not None:
            report["conformal_threshold"] = round(conf_thresh, 4)
            calibration_params["conformal_score_threshold"] = conf_thresh
        report["calibration_params"] = calibration_params
        report["adjustment_applied"] = "platt" if platt else ("conformal" if conf_thresh is not None else "none")

        if not dry_run and (platt or conf_thresh is not None):
            try:
                payload: dict[str, Any] = {
                    "household_id": household_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "last_calibrated_at": datetime.now(timezone.utc).isoformat(),
                }
                if calibration_params:
                    payload["calibration_params"] = calibration_params
                supabase.table("household_calibration").upsert(payload, on_conflict="household_id").execute()
            except Exception as e:
                logger.warning("Calibration upsert failed: %s", e)
        step_trace[-1]["notes"] = report["adjustment_applied"]

    summary = report
    ended_at = datetime.now(timezone.utc).isoformat()
    run_id = persist_agent_run(supabase, household_id, "continual_calibration", started_at=started_at, ended_at=ended_at, status="completed", step_trace=step_trace, summary_json=summary, dry_run=dry_run)
    return {"step_trace": step_trace, "summary_json": summary, "status": "ok", "started_at": started_at, "ended_at": ended_at, "run_id": run_id}
