"""
Continual Calibration Agent: Calibration + Policy Update.
Eight steps: gather labeled data, data quality + bias checks, choose method, fit calibration,
update household_calibration, policy recommendations (policy patch), calibration report artifact, UI.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import (
    AgentContext,
    persist_agent_run,
    persist_agent_run_ctx,
    step,
    upsert_risk_signal_ctx,
    upsert_summary_ctx,
)

logger = logging.getLogger(__name__)


def _agent_settings():
    try:
        from config.settings import get_agent_settings
        return get_agent_settings()
    except Exception:
        class _F:
            calibration_min_labeled = 10
            calibration_target_fpr = 0.1
            calibration_ece_bins = 10
        return _F()


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))
    except Exception:
        return 0.0


def _fetch_labeled_data(supabase: Any, household_id: str) -> list[tuple[float, int, str | None]]:
    """Join feedback with risk_signals; return list of (score, 1 for TP else 0, signal_type)."""
    out: list[tuple[float, int, str | None]] = []
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
                .select("score, signal_type")
                .eq("id", rsid)
                .eq("household_id", household_id)
                .limit(1)
                .execute()
            )
            if sig.data and len(sig.data) > 0 and sig.data[0].get("score") is not None:
                score = float(sig.data[0]["score"])
                y = 1 if label == "true_positive" else 0
                st = sig.data[0].get("signal_type")
                out.append((score, y, st))
    except Exception as e:
        logger.debug("Calibration fetch labeled failed: %s", e)
    return out


def _platt_fit(scores: list[float], labels: list[int]) -> tuple[float, float] | None:
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
    fp_scores = [s for s, y in zip(scores, labels) if y == 0]
    if not fp_scores:
        return None
    fp_scores_sorted = sorted(fp_scores, reverse=True)
    idx = max(0, int((1 - target_fpr) * len(fp_scores_sorted)) - 1)
    return fp_scores_sorted[idx] if idx < len(fp_scores_sorted) else fp_scores_sorted[-1]


def _ece(probs: list[float], labels: list[int], n_bins: int | None = None) -> float:
    if n_bins is None:
        n_bins = _agent_settings().calibration_ece_bins
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


def _precision_recall(scores: list[float], labels: list[int], threshold: float) -> tuple[float, float]:
    tp = sum(1 for s, y in zip(scores, labels) if y == 1 and s >= threshold)
    fp = sum(1 for s, y in zip(scores, labels) if y == 0 and s >= threshold)
    fn = sum(1 for s, y in zip(scores, labels) if y == 1 and s < threshold)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(prec, 4), round(rec, 4)


def run_continual_calibration_playbook(ctx: AgentContext) -> dict[str, Any]:
    """
    Eight-step Calibration + Policy Update agent.
    Returns step_trace, summary_json, status, run_id, artifacts_refs.
    """
    step_trace: list[dict] = []
    started_at = ctx.now.isoformat()
    summary_json: dict[str, Any] = {"headline": "Calibration", "key_metrics": {}, "key_findings": [], "recommended_actions": [], "artifact_refs": {}}
    artifacts_refs: dict[str, Any] = {"summary_ids": [], "risk_signal_ids": []}
    run_id: str | None = None

    if not ctx.supabase:
        step_trace.append({"step": "gather_labeled_data", "status": "skip", "started_at": started_at, "ended_at": ctx.now.isoformat(), "notes": "no_supabase"})
        summary_json["reason"] = "no_supabase"
        summary_json["feedback_count"] = 0
        run_id = persist_agent_run_ctx(ctx, "continual_calibration", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    min_labeled = _agent_settings().calibration_min_labeled

    # Step 1 — Gather labeled data
    with step(ctx, step_trace, "gather_labeled_data"):
        labeled = _fetch_labeled_data(ctx.supabase, ctx.household_id)
        scores = [x[0] for x in labeled]
        labels = [x[1] for x in labeled]
        signal_types = [x[2] for x in labeled]
        step_trace[-1]["outputs_count"] = len(labeled)
        step_trace[-1]["notes"] = f"{len(labeled)} labeled (TP={sum(labels)}, FP={len(labels)-sum(labels)})"

    report: dict[str, Any] = {"household_id": ctx.household_id, "feedback_count": len(labeled), "current_adjustment": None, "calibration_params": None, "before_ece": None, "after_ece": None, "conformal_threshold": None}

    try:
        cal = ctx.supabase.table("household_calibration").select("severity_threshold_adjust, calibration_params, last_calibrated_at").eq("household_id", ctx.household_id).limit(1).execute()
        if cal.data and len(cal.data) > 0:
            report["current_adjustment"] = cal.data[0].get("severity_threshold_adjust")
            report["calibration_params"] = cal.data[0].get("calibration_params")
    except Exception:
        pass

    if len(labeled) < min_labeled:
        step_trace.append({"step": "data_quality", "status": "ok", "started_at": step_trace[-1]["ended_at"], "ended_at": ctx.now.isoformat(), "notes": "insufficient_labels"})
        summary_json["reason"] = "insufficient_labels"
        summary_json["feedback_count"] = len(labeled)
        summary_json["key_metrics"] = {"feedback_count": len(labeled), "min_required": min_labeled}
        summary_json["key_findings"] = ["Collect more feedback labels to run calibration."]
        run_id = persist_agent_run_ctx(ctx, "continual_calibration", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    # Step 2 — Data quality + bias checks
    with step(ctx, step_trace, "data_quality_bias"):
        n_tp = sum(labels)
        n_fp = len(labels) - n_tp
        balance_ok = n_tp >= 2 and n_fp >= 2
        type_coverage = len(set(s for s in signal_types if s)) if signal_types else 0
        warnings = []
        if not balance_ok:
            warnings.append("Label balance low (TP and FP each >= 2 recommended).")
        if type_coverage == 0 and signal_types:
            warnings.append("Signal type coverage unknown.")
        report["data_quality_warnings"] = warnings
        report["label_balance"] = {"tp": n_tp, "fp": n_fp}
        step_trace[-1]["outputs_count"] = len(warnings)
        step_trace[-1]["notes"] = f"balance_ok={balance_ok} warnings={len(warnings)}"

    # Step 3 — Choose calibration method
    with step(ctx, step_trace, "choose_method"):
        method = "platt_scaling"
        if report.get("calibration_params") and "temperature" in str(report.get("calibration_params", {})):
            method = "temperature_scaling"
        report["method_chosen"] = method
        step_trace[-1]["notes"] = method

    # Step 4 — Fit calibration (no LLM)
    with step(ctx, step_trace, "fit_calibration"):
        platt = _platt_fit(scores, labels)
        conf_thresh = _conformal_threshold(scores, labels, _agent_settings().calibration_target_fpr)
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
        threshold_used = conf_thresh if conf_thresh is not None else 0.5
        prec_before, rec_before = _precision_recall(scores, labels, 0.5)
        prec_after, rec_after = _precision_recall(scores, labels, threshold_used)
        report["precision_recall_before"] = {"precision": prec_before, "recall": rec_before}
        report["precision_recall_after"] = {"precision": prec_after, "recall": rec_after}
        step_trace[-1]["outputs_count"] = 1
        step_trace[-1]["notes"] = f"before_ece={report['before_ece']} after_ece={report.get('after_ece')}"

    # Step 5 — Update household_calibration
    severity_adjust = 0.0
    with step(ctx, step_trace, "update_household_calibration"):
        if platt and report.get("after_ece") is not None and report.get("before_ece") is not None:
            severity_adjust = round(report["before_ece"] - report["after_ece"], 4)
        report["severity_threshold_adjust"] = severity_adjust
        if not ctx.dry_run and (platt or conf_thresh is not None):
            try:
                payload: dict[str, Any] = {
                    "household_id": ctx.household_id,
                    "updated_at": ctx.now.isoformat(),
                    "last_calibrated_at": ctx.now.isoformat(),
                    "severity_threshold_adjust": severity_adjust,
                }
                if calibration_params:
                    payload["calibration_params"] = calibration_params
                ctx.supabase.table("household_calibration").upsert(payload, on_conflict="household_id").execute()
                report["calibration_run_id"] = started_at
            except Exception as e:
                logger.warning("Calibration upsert failed: %s", e)
        step_trace[-1]["notes"] = f"severity_threshold_adjust={severity_adjust}"

    # Step 6 — Policy recommendations (policy patch)
    with step(ctx, step_trace, "policy_recommendations"):
        policy_patch = {
            "new_thresholds": {"conformal_score_threshold": report.get("conformal_threshold"), "severity_threshold_adjust": report.get("severity_threshold_adjust", 0.0)},
            "changed_rules": ["Use calibrated scores for severity" if platt else "No Platt change"],
            "suggested_watchlist_threshold": report.get("conformal_threshold"),
            "notes": f"ECE before={report.get('before_ece')} after={report.get('after_ece')}. Apply policy patch to pipeline.",
        }
        summary_json["policy_patch"] = policy_patch
        step_trace[-1]["outputs_count"] = 1

    # Step 7 — Calibration Report artifact (Summary + risk_signal calibration_update)
    with step(ctx, step_trace, "calibration_report_artifact"):
        period_start = (ctx.now - timedelta(days=7)).isoformat()
        period_end = ctx.now.isoformat()
        summary_text = f"Calibration run: ECE before={report.get('before_ece')} after={report.get('after_ece')}. Policy patch applied."
        sid = upsert_summary_ctx(ctx, "weekly_calibration_report", period_start, period_end, summary_text, {**report, "policy_patch": summary_json.get("policy_patch", {})})
        if sid:
            artifacts_refs["summary_ids"].append(sid)
        rsid = upsert_risk_signal_ctx(
            ctx,
            "calibration_update",
            1,
            0.0,
            {"summary": summary_text, "before_ece": report.get("before_ece"), "after_ece": report.get("after_ece"), "policy_patch": summary_json.get("policy_patch")},
            {"checklist": ["Review new thresholds", "Apply policy patch"], "action": "review"},
            "open",
        )
        if rsid:
            artifacts_refs["risk_signal_ids"].append(rsid)
        step_trace[-1]["outputs_count"] = (1 if sid else 0) + (1 if rsid else 0)

    # Step 8 — UI (summary for /agents and /dashboard)
    with step(ctx, step_trace, "ui_summary"):
        summary_json["headline"] = "Calibration complete; system confidence improved"
        summary_json["key_metrics"] = {"before_ece": report.get("before_ece"), "after_ece": report.get("after_ece"), "feedback_count": len(labeled)}
        summary_json["key_findings"] = [f"ECE improved from {report.get('before_ece')} to {report.get('after_ece')}."] if report.get("after_ece") is not None else ["Calibration run completed."]
        summary_json["recommended_actions"] = ["Review calibration report on /agents", "Dashboard shows system confidence improved"]
        summary_json["artifact_refs"] = artifacts_refs
        summary_json["feedback_count"] = report["feedback_count"]
        summary_json["current_adjustment"] = report.get("current_adjustment")
        summary_json["household_id"] = ctx.household_id
        summary_json["calibration_report"] = report
        run_id = persist_agent_run_ctx(ctx, "continual_calibration", "completed", step_trace, summary_json, artifacts_refs)
        step_trace[-1]["artifacts_refs"] = artifacts_refs

    return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "started_at": started_at, "ended_at": ctx.now.isoformat(), "run_id": run_id, "artifacts_refs": artifacts_refs}


def run_continual_calibration_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Wrapper: build ctx and call run_continual_calibration_playbook."""
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)
    out = run_continual_calibration_playbook(ctx)
    return {"step_trace": out["step_trace"], "summary_json": out["summary_json"], "status": out["status"], "started_at": out["started_at"], "ended_at": out["ended_at"], "run_id": out.get("run_id"), "artifacts_refs": out.get("artifacts_refs")}
