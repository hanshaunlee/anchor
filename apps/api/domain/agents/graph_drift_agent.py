"""
Graph Drift Agent: Drift + Root Cause + Action Plan.
Nine steps: intake, data collection, quality checks, drift metrics, slice analysis,
prototype extraction, root-cause classification (optional LLM), action plan + artifacts, persist.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import (
    AgentContext,
    persist_agent_run,
    persist_agent_run_ctx,
    step,
    upsert_risk_signal,
    upsert_risk_signal_ctx,
    upsert_summary,
    upsert_summary_ctx,
)
from domain.ml_artifacts import (
    centroid,
    cosine_sim,
    fetch_embeddings_window,
    compute_mmd_or_energy_distance,
    compute_mmd_rbf,
    compute_drift_confidence_interval,
    normalize,
)

logger = logging.getLogger(__name__)

# Threshold for composite drift decision (exported for tests)
DRIFT_THRESHOLD_TAU = 0.15


def _agent_settings():
    try:
        from config.settings import get_agent_settings
        return get_agent_settings()
    except Exception:
        class _F:
            drift_window_recent_days = 3
            drift_window_baseline_days = 14
            drift_baseline_end_days = 7
            drift_min_samples_per_window = 10
            drift_threshold = 0.15
            drift_top_prototype_examples = 3
            drift_k_neighbors = 5
            drift_pca_components = 10
            drift_mmd_threshold = 0.2
            drift_ks_threshold = 0.3
            drift_severity_high_centroid = 0.25
        return _F()


def _get_model_version_tag() -> str:
    try:
        from config.settings import get_ml_settings
        return getattr(get_ml_settings(), "model_version_tag", "v0") or "v0"
    except Exception:
        return "v0"


def _centroid_shift(recent_emb: list[list[float]], baseline_emb: list[list[float]]) -> float:
    c_r = centroid(recent_emb)
    c_b = centroid(baseline_emb)
    if c_r is None or c_b is None or len(c_r) != len(c_b):
        return 0.0
    return 1.0 - cosine_sim(c_r, c_b)


# Alias for tests that expect _compute_shift (same semantics: 0 when identical, up to 2 when opposite).
_compute_shift = _centroid_shift


def _embedding_norm_drift(baseline_emb: list[list[float]], recent_emb: list[list[float]]) -> float:
    def mean_norm(embeddings: list[list[float]]) -> float:
        if not embeddings:
            return 0.0
        n = sum((sum(x * x for x in e) ** 0.5) for e in embeddings)
        return n / len(embeddings)
    return abs(mean_norm(recent_emb) - mean_norm(baseline_emb))


def _ks_aggregate(baseline_emb: list[list[float]], recent_emb: list[list[float]]) -> float:
    if not baseline_emb or not recent_emb or len(baseline_emb[0]) < 2:
        return 0.0
    pca_components = _agent_settings().drift_pca_components
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        X_b = np.array(baseline_emb, dtype=float)
        X_r = np.array(recent_emb, dtype=float)
        n_comp = min(pca_components, X_b.shape[1], X_b.shape[0] - 1)
        if n_comp < 1:
            return 0.0
        pca = PCA(n_components=n_comp).fit(X_b)
        B = pca.transform(X_b)
        R = pca.transform(X_r)
        from scipy import stats
        ks_vals = [stats.ks_2samp(B[:, j].tolist(), R[:, j].tolist())[0] for j in range(n_comp)]
        return float(max(ks_vals)) if ks_vals else 0.0
    except ImportError:
        return 0.0
    except Exception as e:
        logger.debug("KS aggregate failed: %s", e)
        return 0.0


def _neighbor_stability(
    baseline_emb: list[list[float]],
    recent_emb: list[list[float]],
    k: int | None = None,
) -> float:
    if k is None:
        k = _agent_settings().drift_k_neighbors
    if len(baseline_emb) < k + 1 or len(recent_emb) < k + 1:
        return 0.0
    def avg_top_k_sim(embeddings: list[list[float]]) -> float:
        total, count = 0.0, 0
        for i, e in enumerate(embeddings):
            sims = [cosine_sim(e, embeddings[j]) for j in range(len(embeddings)) if j != i]
            sims.sort(reverse=True)
            top = sims[:k]
            if top:
                total += sum(top) / len(top)
                count += 1
        return total / count if count else 0.0
    return abs(avg_top_k_sim(recent_emb) - avg_top_k_sim(baseline_emb))


def _quality_report(baseline_rows: list[dict], recent_rows: list[dict]) -> dict[str, Any]:
    """Dim consistency, missing/NaNs, L2 norms distribution."""
    all_emb = [r.get("embedding") for r in baseline_rows + recent_rows if r.get("embedding")]
    dims = [len(e) for e in all_emb]
    dim_consistent = len(set(dims)) <= 1 and len(dims) == len(all_emb)
    uniq_dim = dims[0] if dims else 0
    norms = []
    has_nan = False
    for e in all_emb:
        n = sum(x * x for x in e) ** 0.5
        norms.append(n)
        if any(x != x for x in e):
            has_nan = True
    import statistics
    norm_mean = statistics.mean(norms) if norms else 0.0
    norm_stdev = statistics.stdev(norms) if len(norms) > 1 else 0.0
    return {
        "dim_consistent": dim_consistent,
        "dim": uniq_dim,
        "n_embeddings": len(all_emb),
        "has_nan": has_nan,
        "l2_norm_mean": round(norm_mean, 4),
        "l2_norm_stdev": round(norm_stdev, 4),
    }


def _slice_drift(
    baseline_rows: list[dict],
    recent_rows: list[dict],
    supabase: Any,
    household_id: str,
) -> list[dict]:
    """Drift per signal_type and severity bucket; top 3 slices with counts and example signal_ids."""
    def get_signal_meta(rid: str) -> tuple[str, int]:
        try:
            r = supabase.table("risk_signals").select("signal_type, severity").eq("id", rid).eq("household_id", household_id).limit(1).execute()
            if r.data and r.data[0]:
                return (r.data[0].get("signal_type") or "unknown", int(r.data[0].get("severity") or 0))
        except Exception:
            pass
        return "unknown", 0

    baseline_meta = [get_signal_meta(r.get("risk_signal_id") or "") for r in baseline_rows if r.get("risk_signal_id")]
    recent_meta = [get_signal_meta(r.get("risk_signal_id") or "") for r in recent_rows if r.get("risk_signal_id")]
    from collections import Counter
    b_type = Counter(t for t, _ in baseline_meta)
    r_type = Counter(t for t, _ in recent_meta)
    b_sev = Counter(s for _, s in baseline_meta)
    r_sev = Counter(s for _, s in recent_meta)
    slices = []
    for st in set(b_type) | set(r_type):
        cb, cr = b_type.get(st, 0), r_type.get(st, 0)
        delta = abs(cr - cb) / max(cb + cr, 1)
        example_ids = [r.get("risk_signal_id") for r in recent_rows if get_signal_meta(r.get("risk_signal_id") or "")[0] == st][:3]
        slices.append({"slice": f"signal_type:{st}", "baseline_count": cb, "recent_count": cr, "drift_delta": round(delta, 4), "example_risk_signal_ids": [e for e in example_ids if e]})
    for sev in set(b_sev) | set(r_sev):
        cb, cr = b_sev.get(sev, 0), r_sev.get(sev, 0)
        delta = abs(cr - cb) / max(cb + cr, 1)
        example_ids = [r.get("risk_signal_id") for r in recent_rows if get_signal_meta(r.get("risk_signal_id") or "")[1] == sev][:3]
        slices.append({"slice": f"severity:{sev}", "baseline_count": cb, "recent_count": cr, "drift_delta": round(delta, 4), "example_risk_signal_ids": [e for e in example_ids if e]})
    slices.sort(key=lambda x: -x["drift_delta"])
    return slices[:3]


def _prototype_incidents(
    recent_rows: list[dict],
    baseline_centroid_vec: list[float],
) -> list[dict]:
    """3 representative: farthest from baseline centroid + 2 near centroid (typical)."""
    if not baseline_centroid_vec or not recent_rows:
        return []
    scored = []
    for row in recent_rows:
        emb = row.get("embedding")
        rid = row.get("risk_signal_id")
        if not emb or not rid or len(emb) != len(baseline_centroid_vec):
            continue
        sim = cosine_sim(emb, baseline_centroid_vec)
        scored.append({"risk_signal_id": rid, "distance_from_baseline": round(1.0 - sim, 4), "summary": "farthest" if len(scored) == 0 else "typical"})
    scored.sort(key=lambda x: -x["distance_from_baseline"])
    top_n = _agent_settings().drift_top_prototype_examples
    out = []
    if scored:
        out.append({**scored[0], "summary": "farthest from baseline centroid"})
    for s in scored[1:top_n]:
        out.append({**s, "summary": "typical recent"})
    return out[:top_n]


def _root_cause_llm(metrics: dict, slices: list, prototypes: list, model_meta: dict) -> dict | None:
    """Optional LangChain structured RootCause; fallback None."""
    try:
        from pydantic import BaseModel, Field
        from domain.langchain_utils import get_llm, run_structured_prompt

        class RootCause(BaseModel):
            cause: str = Field(description="One of: behavior_shift, model_change, data_pipeline_change, new_scam_theme, seasonality, unknown")
            confidence: float = Field(ge=0, le=1)
            rationale: str
            recommended_actions: list[str] = Field(default_factory=list)

        llm = get_llm()
        if not llm:
            return None
        prompt = (
            "Given drift metrics and slice analysis, classify the root cause of embedding distribution shift. "
            "Metrics: " + str(metrics) + ". Top slices: " + str(slices[:2]) + ". "
            "Respond with cause (one of behavior_shift, model_change, data_pipeline_change, new_scam_theme, seasonality, unknown), confidence 0-1, rationale, and recommended_actions list."
        )
        result = run_structured_prompt(llm, prompt, RootCause)
        if result:
            return {"cause": result.cause, "confidence": result.confidence, "rationale": result.rationale, "recommended_actions": result.recommended_actions or []}
    except Exception as e:
        logger.debug("Root cause LLM failed: %s", e)
    return None


def _cause_label_deterministic(model_tag_changed: bool, new_signal_types: bool) -> str:
    if model_tag_changed:
        return "model_change"
    if new_signal_types:
        return "new_pattern"
    return "behavior_shift"


def run_graph_drift_playbook(
    ctx: AgentContext,
    *,
    window_recent_days: int = 3,
    window_baseline_days: int = 14,
    min_samples: int = 10,
    embedding_space: str = "risk_signal",
) -> dict[str, Any]:
    """
    Nine-step Drift + Root Cause + Action Plan agent.
    Returns step_trace, summary_json, status, run_id, artifacts_refs.
    """
    cfg = _agent_settings()
    step_trace: list[dict] = []
    started_at = ctx.now.isoformat()
    summary_json: dict[str, Any] = {}
    artifacts_refs: dict[str, Any] = {"risk_signal_ids": [], "summary_ids": []}
    run_id: str | None = None

    # Step 1 — Intake & scope
    with step(ctx, step_trace, "intake_scope", notes=f"recent={window_recent_days}d baseline={window_baseline_days}d min_samples={min_samples}"):
        now = ctx.now
        baseline_end_days = getattr(cfg, "drift_baseline_end_days", 7)
        baseline_start = (now - timedelta(days=window_baseline_days)).isoformat()
        baseline_end = (now - timedelta(days=baseline_end_days)).isoformat()
        recent_start = (now - timedelta(days=window_recent_days)).isoformat()
        recent_end = now.isoformat()
        model_tag = _get_model_version_tag()
        scope = {"household_id": ctx.household_id, "embedding_space": embedding_space, "consent_keys": list(ctx.consent_state.keys())}
        step_trace[-1]["artifacts_refs"] = scope

    if not ctx.supabase:
        step_trace.append({"step": "data_collection", "status": "skip", "started_at": step_trace[-1]["ended_at"], "ended_at": ctx.now.isoformat(), "notes": "no_supabase"})
        summary_json = {"headline": "Drift check skipped", "reason": "no_supabase", "key_metrics": {}, "key_findings": [], "recommended_actions": ["Configure Supabase to run drift detection."], "artifact_refs": {}}
        run_id = persist_agent_run_ctx(ctx, "graph_drift", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "artifacts_refs": artifacts_refs, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    # Step 2 — Data collection
    with step(ctx, step_trace, "data_collection", inputs_count=None):
        baseline_rows = fetch_embeddings_window(
            ctx.supabase, ctx.household_id, (baseline_start, baseline_end),
            require_has_embedding=True, embedding_space=embedding_space,
        )
        recent_rows = fetch_embeddings_window(
            ctx.supabase, ctx.household_id, (recent_start, recent_end),
            require_has_embedding=True, embedding_space=embedding_space,
        )
        baseline_emb = [r["embedding"] for r in baseline_rows if r.get("embedding")]
        recent_emb = [r["embedding"] for r in recent_rows if r.get("embedding")]
        step_trace[-1]["inputs_count"] = len(baseline_rows) + len(recent_rows)
        step_trace[-1]["outputs_count"] = len(baseline_emb) + len(recent_emb)
        step_trace[-1]["notes"] = f"baseline={len(baseline_emb)} recent={len(recent_emb)}"
        model_meta = {"checkpoint_id": None, "model_version_tag": model_tag}
        for r in baseline_rows + recent_rows:
            if r.get("checkpoint_id") or r.get("model_name"):
                model_meta["checkpoint_id"] = r.get("checkpoint_id") or r.get("model_name")
                break

    if len(baseline_emb) < min_samples or len(recent_emb) < min_samples:
        step_trace.append({"step": "quality_checks", "status": "ok", "started_at": step_trace[-1]["ended_at"], "ended_at": ctx.now.isoformat(), "notes": "insufficient_samples"})
        summary_json = {
            "headline": "Insufficient samples for drift analysis",
            "reason": "insufficient_samples",
            "key_metrics": {"n_baseline": len(baseline_emb), "n_recent": len(recent_emb), "min_required": min_samples},
            "key_findings": ["Collect more risk signals with embeddings in baseline and recent windows."],
            "recommended_actions": ["Collect more risk signals with embeddings in baseline and recent windows."],
            "artifact_refs": {},
        }
        run_id = persist_agent_run_ctx(ctx, "graph_drift", "completed", step_trace, summary_json, artifacts_refs)
        return {"step_trace": step_trace, "summary_json": summary_json, "status": "ok", "run_id": run_id, "artifacts_refs": artifacts_refs, "started_at": started_at, "ended_at": ctx.now.isoformat()}

    # Step 3 — Quality checks
    with step(ctx, step_trace, "quality_checks"):
        quality_report = _quality_report(baseline_rows, recent_rows)
        summary_json["quality_report"] = quality_report
        step_trace[-1]["outputs_count"] = 1
        step_trace[-1]["notes"] = f"dim_consistent={quality_report.get('dim_consistent')}"

    # Step 4 — Drift metrics (no LLM). Drift refers to embedding distribution shift relative to historical baseline.
    with step(ctx, step_trace, "drift_metrics"):
        centroid_shift = round(_centroid_shift(recent_emb, baseline_emb), 4)
        mmd = round(compute_mmd_or_energy_distance(baseline_emb, recent_emb, use_energy=True), 4)
        mmd_rbf = round(compute_mmd_rbf(baseline_emb, recent_emb), 4)  # MMD with RBF kernel; bandwidth = median heuristic
        ks = round(_ks_aggregate(baseline_emb, recent_emb), 4)
        norm_drift = round(_embedding_norm_drift(baseline_emb, recent_emb), 4)
        neighbor_stability_delta = round(_neighbor_stability(baseline_emb, recent_emb), 4)
        point, ci_lo, ci_hi = compute_drift_confidence_interval(
            baseline_emb, recent_emb, lambda b, r: _centroid_shift(r, b), n_bootstrap=100, confidence=0.95,
        )
        drift_confidence_interval = [round(ci_lo, 4), round(ci_hi, 4)]
        metrics = {
            "centroid_shift": centroid_shift,
            "mmd": mmd,
            "mmd_rbf": mmd_rbf,
            "ks_stat": ks,
            "pca_ks_aggregate": ks,
            "norm_drift": norm_drift,
            "neighbor_stability_delta": neighbor_stability_delta,
            "drift_confidence_interval": drift_confidence_interval,
        }
        summary_json["key_metrics"] = metrics
        step_trace[-1]["outputs_count"] = len(metrics)
        step_trace[-1]["notes"] = f"centroid_shift={centroid_shift} mmd={mmd} mmd_rbf={mmd_rbf}"

    # Step 5 — Slice analysis
    with step(ctx, step_trace, "slice_analysis"):
        top_slices = _slice_drift(baseline_rows, recent_rows, ctx.supabase, ctx.household_id)
        summary_json["top_slices"] = top_slices
        step_trace[-1]["outputs_count"] = len(top_slices)

    # Step 6 — Prototype extraction
    with step(ctx, step_trace, "prototype_extraction"):
        c_b = centroid(baseline_emb)
        prototypes = _prototype_incidents(recent_rows, c_b or [])
        summary_json["prototypes"] = prototypes
        step_trace[-1]["outputs_count"] = len(prototypes)
        step_trace[-1]["artifacts_refs"] = [p.get("risk_signal_id") for p in prototypes if p.get("risk_signal_id")]

    # Step 7 — Root cause classification (LangChain optional)
    drift_threshold = getattr(cfg, "drift_threshold", DRIFT_THRESHOLD_TAU)
    mmd_t = getattr(cfg, "drift_mmd_threshold", 0.2)
    ks_t = getattr(cfg, "drift_ks_threshold", 0.3)
    drift_detected = metrics["centroid_shift"] > drift_threshold or (mmd > mmd_t and ks > ks_t)

    with step(ctx, step_trace, "root_cause_classification"):
        root_cause_result = _root_cause_llm(metrics, top_slices, prototypes, model_meta)
        if root_cause_result:
            cause = root_cause_result.get("cause", "unknown")
            rationale = root_cause_result.get("rationale", "")
            recommended_actions = root_cause_result.get("recommended_actions", [])
        else:
            model_tag_changed = False
            try:
                checkpoints = {r.get("checkpoint_id") or r.get("model_name") for r in (baseline_rows + recent_rows) if r.get("checkpoint_id") or r.get("model_name")}
                model_tag_changed = len(checkpoints) > 1
            except Exception:
                pass
            signal_types_baseline = set()
            signal_types_recent = set()
            try:
                for r in baseline_rows:
                    mid = r.get("risk_signal_id")
                    if mid:
                        sig = ctx.supabase.table("risk_signals").select("signal_type").eq("id", mid).eq("household_id", ctx.household_id).limit(1).execute()
                        if sig.data and sig.data[0]:
                            signal_types_baseline.add(sig.data[0].get("signal_type", ""))
                for r in recent_rows:
                    mid = r.get("risk_signal_id")
                    if mid:
                        sig = ctx.supabase.table("risk_signals").select("signal_type").eq("id", mid).eq("household_id", ctx.household_id).limit(1).execute()
                        if sig.data and sig.data[0]:
                            signal_types_recent.add(sig.data[0].get("signal_type", ""))
            except Exception:
                pass
            cause = _cause_label_deterministic(model_tag_changed, bool(signal_types_recent - signal_types_baseline))
            rationale = f"Deterministic: model_change={model_tag_changed}, new_signal_types={bool(signal_types_recent - signal_types_baseline)}"
            recommended_actions = []
        summary_json["cause"] = cause
        summary_json["rationale"] = rationale
        summary_json["key_findings"] = [f"Root cause: {cause}. {rationale}"]
        if recommended_actions:
            summary_json["recommended_actions"] = recommended_actions
        step_trace[-1]["notes"] = cause

    # Step 8 — Action plan + artifacts
    with step(ctx, step_trace, "action_plan_artifacts"):
        if not summary_json.get("recommended_actions"):
            summary_json["recommended_actions"] = [
                "run synthetic_redteam regression",
                "retrain model",
                "tighten threshold temporarily",
                "review top prototype incidents",
            ]
        recommended_action_json = {"checklist": summary_json["recommended_actions"], "retrain_suggested": True, "action": "review"}
        risk_signal_id = None
        summary_id = None
        if drift_detected:
            severity = 3 if metrics["centroid_shift"] > getattr(cfg, "drift_severity_high_centroid", 0.25) else 2
            explanation = {
                "summary": f"Embedding drift detected: centroid_shift={metrics['centroid_shift']:.3f}, MMD={mmd:.3f}. Cause: {cause}.",
                "metrics": metrics,
                "cause": cause,
                "example_risk_signal_ids": [p.get("risk_signal_id") for p in prototypes if p.get("risk_signal_id")],
                "model_available": True,
            }
            risk_signal_id = upsert_risk_signal_ctx(ctx, "drift_warning", severity, float(metrics["centroid_shift"]), explanation, recommended_action_json, "open")
            if risk_signal_id:
                artifacts_refs["risk_signal_ids"].append(risk_signal_id)
            # Drift invalidates conformal: exchangeability may no longer hold. Mark conformal stale so pipeline/worker do not use q_hat until recalibration.
            if not ctx.dry_run and ctx.supabase:
                try:
                    cal_r = ctx.supabase.table("household_calibration").select("calibration_params").eq("household_id", ctx.household_id).limit(1).execute()
                    params = {}
                    if cal_r.data and len(cal_r.data) > 0 and cal_r.data[0].get("calibration_params"):
                        params = dict(cal_r.data[0]["calibration_params"])
                    params["conformal_invalid_since"] = ctx.now.isoformat()
                    params.pop("conformal_q_hat", None)
                    params.pop("coverage_level", None)
                    params.pop("calibration_size", None)
                    ctx.supabase.table("household_calibration").upsert({
                        "household_id": ctx.household_id,
                        "updated_at": ctx.now.isoformat(),
                        "calibration_params": params,
                    }, on_conflict="household_id").execute()
                except Exception as ex:
                    logger.debug("Drift: invalidate conformal failed: %s", ex)
            summary_id = upsert_summary_ctx(
                ctx,
                "drift_report",
                recent_start,
                recent_end,
                f"Weekly drift check: Drift detected. {cause}. Review prototypes.",
                summary_json,
            )
            if summary_id:
                artifacts_refs["summary_ids"].append(summary_id)
        else:
            summary_json["recommended_actions"] = ["No significant drift; continue monitoring."]
            summary_id = upsert_summary_ctx(ctx, "drift_report", recent_start, recent_end, "Weekly drift check: No significant drift.", summary_json)
            if summary_id:
                artifacts_refs["summary_ids"].append(summary_id)
        summary_json["drift_detected"] = drift_detected
        summary_json["headline"] = "Drift detected; review prototypes" if drift_detected else "No significant drift"
        summary_json["artifact_refs"] = artifacts_refs
        # Copy-pasteable Modal retrain command for UI "Copy retrain command" (when drift detected).
        if drift_detected:
            summary_json["retrain_command"] = "modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml"
        else:
            summary_json["retrain_command"] = None
        step_trace[-1]["outputs_count"] = (1 if risk_signal_id else 0) + (1 if summary_id else 0)
        step_trace[-1]["artifacts_refs"] = artifacts_refs

    # Step 9 — Persist & notify
    with step(ctx, step_trace, "persist_notify"):
        run_id = persist_agent_run_ctx(ctx, "graph_drift", "completed", step_trace, summary_json, artifacts_refs)
        step_trace[-1]["artifacts_refs"] = {"run_id": run_id, **artifacts_refs}

    ended_at = ctx.now.isoformat()
    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ended_at,
        "run_id": run_id,
        "artifacts_refs": artifacts_refs,
    }


def run_graph_drift_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
    *,
    window_recent_days: int | None = None,
    window_baseline_days: int | None = None,
    baseline_end_days: int | None = None,
    min_samples_per_window: int | None = None,
    drift_threshold: float | None = None,
    embedding_space: str = "risk_signal",
) -> dict[str, Any]:
    """Thin wrapper: build ctx and call run_graph_drift_playbook. Keeps router compatibility."""
    cfg = _agent_settings()
    window_recent_days = window_recent_days if window_recent_days is not None else cfg.drift_window_recent_days
    window_baseline_days = window_baseline_days if window_baseline_days is not None else cfg.drift_window_baseline_days
    min_samples = min_samples_per_window if min_samples_per_window is not None else cfg.drift_min_samples_per_window
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)
    out = run_graph_drift_playbook(ctx, window_recent_days=window_recent_days, window_baseline_days=window_baseline_days, min_samples=min_samples, embedding_space=embedding_space)
    return {
        "step_trace": out["step_trace"],
        "summary_json": out["summary_json"],
        "status": out["status"],
        "started_at": out["started_at"],
        "ended_at": out["ended_at"],
        "run_id": out.get("run_id"),
    }
