"""
Model Health Agent: unified orchestration for drift, calibration, conformal validity, and optional redteam.
Used by Supervisor NIGHTLY_MAINTENANCE; also runnable via Admin Tools.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run_ctx, step

logger = logging.getLogger(__name__)


def _fetch_calibration_meta(supabase: Any, household_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"calibration_params": None, "last_calibrated_at": None, "stale": False}
    if not supabase:
        return out
    try:
        r = (
            supabase.table("household_calibration")
            .select("calibration_params, last_calibrated_at, meta")
            .eq("household_id", household_id)
            .limit(1)
            .execute()
        )
        if r.data and len(r.data) > 0:
            row = r.data[0]
            out["calibration_params"] = row.get("calibration_params")
            out["last_calibrated_at"] = row.get("last_calibrated_at")
            meta = row.get("meta") or {}
            out["stale"] = meta.get("calibration_stale", False)
    except Exception as e:
        logger.debug("Fetch calibration meta failed: %s", e)
    return out


def run_model_health_agent(
    household_id: str,
    supabase: Any | None = None,
    *,
    dry_run: bool = False,
    env: str = "prod",
    admin_force: bool = False,
    run_redteam: bool | None = None,
) -> dict[str, Any]:
    """
    Run drift check, calibration check, conformal validity, and optionally redteam.
    Returns step_trace, summary_json with drift_detected, mmd_rbf, ks_stat, calibration_ece, conformal_coverage, recommendation.
    Redteam runs only if env != "prod" or admin_force=true or run_redteam=true.
    """
    from domain.agents.base import AgentContext

    ctx = AgentContext(
        household_id=household_id,
        supabase=supabase,
        dry_run=dry_run,
    )
    step_trace: list[dict] = []
    started_at = ctx.now.isoformat()
    summary_json: dict[str, Any] = {
        "headline": "Model Health",
        "drift_detected": False,
        "mmd_rbf": None,
        "ks_stat": None,
        "drift_confidence_interval": None,
        "calibration_ece": None,
        "conformal_coverage": None,
        "conformal_q_hat": None,
        "recommendation": "stable",
        "redteam_run": False,
        "warnings": [],
    }
    child_run_ids: dict[str, str | None] = {"drift": None, "calibration": None, "redteam": None}

    # Step 1 — Gather artifacts (calibration meta, embedding availability)
    with step(ctx, step_trace, "gather_artifacts"):
        cal_meta = _fetch_calibration_meta(ctx.supabase or None, household_id)
        summary_json["calibration_params_present"] = bool(cal_meta.get("calibration_params"))
        summary_json["calibration_stale"] = cal_meta.get("stale", False)
        step_trace[-1]["notes"] = f"calibration_present={summary_json['calibration_params_present']} stale={summary_json['calibration_stale']}"

    # Step 2 — Drift check (run graph_drift agent)
    with step(ctx, step_trace, "drift_check"):
        try:
            from domain.agents.graph_drift_agent import run_graph_drift_agent
            drift_result = run_graph_drift_agent(household_id, supabase=supabase, dry_run=dry_run)
            child_run_ids["drift"] = drift_result.get("run_id")
            sj = drift_result.get("summary_json") or {}
            km = sj.get("key_metrics") or {}
            summary_json["drift_detected"] = sj.get("drift_detected", False)
            summary_json["mmd_rbf"] = sj.get("mmd_rbf") or km.get("mmd_rbf")
            summary_json["ks_stat"] = sj.get("ks_stat") or km.get("ks_stat")
            summary_json["drift_confidence_interval"] = sj.get("drift_confidence_interval") or km.get("drift_confidence_interval")
            step_trace[-1]["outputs_count"] = 1
            step_trace[-1]["notes"] = f"drift_detected={summary_json['drift_detected']}"
        except Exception as e:
            logger.exception("Drift check failed: %s", e)
            step_trace[-1]["status"] = "error"
            step_trace[-1]["error"] = str(e)
            summary_json["warnings"].append(f"drift_check_error: {e}")

    # Step 3 — Calibration check (run continual_calibration agent)
    with step(ctx, step_trace, "calibration_check"):
        try:
            from domain.agents.continual_calibration_agent import run_continual_calibration_agent
            cal_result = run_continual_calibration_agent(household_id, supabase=supabase, dry_run=dry_run)
            child_run_ids["calibration"] = cal_result.get("run_id")
            sj = cal_result.get("summary_json") or {}
            km = sj.get("key_metrics") or {}
            report = sj.get("calibration_report") or {}
            summary_json["calibration_ece"] = sj.get("after_ece") or km.get("after_ece") or report.get("after_ece")
            summary_json["conformal_coverage"] = sj.get("coverage_level") or report.get("coverage_level_requested")
            summary_json["conformal_q_hat"] = sj.get("conformal_q_hat") or report.get("conformal_q_hat")
            step_trace[-1]["outputs_count"] = 1
        except Exception as e:
            logger.exception("Calibration check failed: %s", e)
            step_trace[-1]["status"] = "error"
            step_trace[-1]["error"] = str(e)
            summary_json["warnings"].append(f"calibration_check_error: {e}")

    # Step 4 — Conformal validity (ensure calibration window valid; mark stale if drift severe)
    with step(ctx, step_trace, "conformal_validity_check"):
        if summary_json.get("drift_detected") and ctx.supabase and not dry_run:
            try:
                # Mark calibration stale when drift is severe so escalation logic can back off
                r = ctx.supabase.table("household_calibration").select("id, meta").eq("household_id", household_id).limit(1).execute()
                if r.data and len(r.data) > 0:
                    meta = (r.data[0].get("meta") or {}) if isinstance(r.data[0].get("meta"), dict) else {}
                    meta["calibration_stale"] = True
                    ctx.supabase.table("household_calibration").update({"meta": meta}).eq("household_id", household_id).execute()
                    summary_json["calibration_stale"] = True
                    summary_json["recommendation"] = "recalibrate"
            except Exception as e:
                logger.debug("Mark calibration stale failed: %s", e)
        step_trace[-1]["notes"] = f"recommendation={summary_json['recommendation']}"

    # Step 5 — Redteam regression (optional; skip in prod unless admin_force)
    do_redteam = run_redteam if run_redteam is not None else (env != "prod" or admin_force)
    if do_redteam:
        with step(ctx, step_trace, "redteam_regression"):
            try:
                from domain.agents.synthetic_redteam_agent import run_synthetic_redteam_agent
                red_result = run_synthetic_redteam_agent(household_id, supabase=supabase, dry_run=dry_run)
                child_run_ids["redteam"] = red_result.get("run_id")
                summary_json["redteam_run"] = True
                sj = red_result.get("summary_json") or {}
                summary_json["redteam_pass_rate"] = sj.get("pass_rate")
                step_trace[-1]["outputs_count"] = 1
            except Exception as e:
                logger.exception("Redteam failed: %s", e)
                step_trace[-1]["status"] = "error"
                step_trace[-1]["error"] = str(e)
                summary_json["warnings"].append(f"redteam_error: {e}")

    # Step 6 — Synthesis report
    with step(ctx, step_trace, "synthesis_report"):
        if summary_json.get("drift_detected"):
            if summary_json.get("recommendation") != "recalibrate":
                summary_json["recommendation"] = "retrain"
        if summary_json.get("calibration_stale"):
            summary_json["recommendation"] = "recalibrate"
        summary_json["child_run_ids"] = child_run_ids
        step_trace[-1]["notes"] = summary_json["recommendation"]

    run_id = persist_agent_run_ctx(ctx, "model_health", "completed", step_trace, summary_json)
    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "status": "ok",
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ctx.now.isoformat(),
        "child_run_ids": child_run_ids,
    }
