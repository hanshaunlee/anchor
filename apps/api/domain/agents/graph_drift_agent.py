"""
Graph Drift Agent: multi-metric embedding distribution shift with root-cause and actions.
Research-grade: centroid shift, MMD/energy distance, PCA+KS, norm drift, neighbor stability.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run, step, upsert_risk_signal, upsert_summary
from domain.ml_artifacts import (
    centroid,
    cosine_sim,
    fetch_embeddings_window,
    compute_mmd_or_energy_distance,
    normalize,
)

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_RECENT_DAYS = 3
DEFAULT_WINDOW_BASELINE_DAYS = 14
DEFAULT_BASELINE_END_DAYS = 7
MIN_SAMPLES_PER_WINDOW = 10
DRIFT_THRESHOLD = 0.15
DRIFT_THRESHOLD_TAU = DRIFT_THRESHOLD  # backward compat
TOP_PROTOTYPE_EXAMPLES = 3
K_NEIGHBORS = 5
PCA_COMPONENTS = 10


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


def _embedding_norm_drift(baseline_emb: list[list[float]], recent_emb: list[list[float]]) -> float:
    def mean_norm(embeddings: list[list[float]]) -> float:
        if not embeddings:
            return 0.0
        n = sum((sum(x * x for x in e) ** 0.5) for e in embeddings)
        return n / len(embeddings)
    return abs(mean_norm(recent_emb) - mean_norm(baseline_emb))


def _ks_aggregate(baseline_emb: list[list[float]], recent_emb: list[list[float]]) -> float:
    """PCA(10) on baseline, project both; per-dim KS stats; aggregate (max or mean)."""
    if not baseline_emb or not recent_emb or len(baseline_emb[0]) < 2:
        return 0.0
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        X_b = np.array(baseline_emb, dtype=float)
        X_r = np.array(recent_emb, dtype=float)
        n_comp = min(PCA_COMPONENTS, X_b.shape[1], X_b.shape[0] - 1)
        if n_comp < 1:
            return 0.0
        pca = PCA(n_components=n_comp).fit(X_b)
        B = pca.transform(X_b)
        R = pca.transform(X_r)
        from scipy import stats
        ks_vals = []
        for j in range(n_comp):
            ks_stat, _ = stats.ks_2samp(B[:, j].tolist(), R[:, j].tolist())
            ks_vals.append(ks_stat)
        return float(max(ks_vals)) if ks_vals else 0.0
    except ImportError:
        return 0.0
    except Exception as e:
        logger.debug("KS aggregate failed: %s", e)
        return 0.0


def _neighbor_stability(
    baseline_emb: list[list[float]],
    recent_emb: list[list[float]],
    k: int = K_NEIGHBORS,
) -> float:
    """Avg top-k neighbor similarity within baseline vs within recent; return delta (higher = less stable)."""
    if len(baseline_emb) < k + 1 or len(recent_emb) < k + 1:
        return 0.0
    def avg_top_k_sim(embeddings: list[list[float]]) -> float:
        total = 0.0
        count = 0
        for i, e in enumerate(embeddings):
            sims = [cosine_sim(e, embeddings[j]) for j in range(len(embeddings)) if j != i]
            sims.sort(reverse=True)
            top = sims[:k]
            if top:
                total += sum(top) / len(top)
                count += 1
        return total / count if count else 0.0
    s_b = avg_top_k_sim(baseline_emb)
    s_r = avg_top_k_sim(recent_emb)
    return abs(s_r - s_b)


def _prototype_incidents(
    recent_rows: list[dict],
    baseline_centroid_vec: list[float],
) -> list[str]:
    """Return risk_signal_ids of TOP_PROTOTYPE_EXAMPLES recent embeddings farthest from baseline centroid."""
    if not baseline_centroid_vec or not recent_rows:
        return []
    scored = []
    for row in recent_rows:
        emb = row.get("embedding")
        rid = row.get("risk_signal_id")
        if not emb or not rid or len(emb) != len(baseline_centroid_vec):
            continue
        sim = cosine_sim(emb, baseline_centroid_vec)
        scored.append((rid, 1.0 - sim))
    scored.sort(key=lambda x: -x[1])
    return [rid for rid, _ in scored[:TOP_PROTOTYPE_EXAMPLES]]


def _cause_label(
    model_tag_changed: bool,
    new_signal_types: bool,
) -> str:
    if model_tag_changed:
        return "model_change"
    if new_signal_types:
        return "new_pattern"
    return "behavior_shift"


def run_graph_drift_agent(
    household_id: str,
    supabase: Any | None = None,
    dry_run: bool = False,
    *,
    window_recent_days: int = DEFAULT_WINDOW_RECENT_DAYS,
    window_baseline_days: int = DEFAULT_WINDOW_BASELINE_DAYS,
    baseline_end_days: int = DEFAULT_BASELINE_END_DAYS,
    min_samples_per_window: int = MIN_SAMPLES_PER_WINDOW,
    drift_threshold: float = DRIFT_THRESHOLD,
    embedding_space: str = "risk_signal",
) -> dict[str, Any]:
    """
    Multi-metric drift detector. Persists agent_run with summary_json; if drift_detected
    creates risk_signal drift_warning and optional summary. No fake outputs; insufficient
    samples produce report only with explicit reason.
    """
    step_trace: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    ctx = AgentContext(household_id, supabase, dry_run=dry_run)

    # Step 0: config
    with step(ctx, step_trace, "inputs_config"):
        now = ctx.now
        baseline_start = (now - timedelta(days=window_baseline_days)).isoformat()
        baseline_end_dt = now - timedelta(days=baseline_end_days)
        baseline_end = baseline_end_dt.isoformat()
        recent_start = (now - timedelta(days=window_recent_days)).isoformat()
        recent_end = now.isoformat()
        model_tag = _get_model_version_tag()
        step_trace[-1]["notes"] = f"recent={window_recent_days}d baseline={window_baseline_days}d threshold={drift_threshold}"

    if not supabase:
        step_trace.append({
            "step": "fetch_embeddings",
            "status": "skip",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": "no_supabase",
        })
        summary = {
            "metrics": {},
            "drift_detected": False,
            "cause": None,
            "examples": [],
            "per_type": {},
            "recommendations": ["Configure Supabase to run drift detection."],
            "reason": "no_supabase",
        }
        ended_at = datetime.now(timezone.utc).isoformat()
        run_id = persist_agent_run(
            supabase, household_id, "graph_drift",
            started_at=started_at, ended_at=ended_at, status="completed",
            step_trace=step_trace, summary_json=summary, dry_run=dry_run,
        ) if supabase else None
        return {
            "step_trace": step_trace,
            "summary_json": summary,
            "status": "ok",
            "started_at": started_at,
            "ended_at": ended_at,
            "run_id": run_id,
        }

    # Step 1: fetch embeddings
    with step(ctx, step_trace, "fetch_embeddings"):
        baseline_rows = fetch_embeddings_window(
            supabase, household_id, (baseline_start, baseline_end),
            require_has_embedding=True, embedding_space=embedding_space,
        )
        recent_rows = fetch_embeddings_window(
            supabase, household_id, (recent_start, recent_end),
            require_has_embedding=True, embedding_space=embedding_space,
        )
        baseline_emb = [r["embedding"] for r in baseline_rows if r.get("embedding")]
        recent_emb = [r["embedding"] for r in recent_rows if r.get("embedding")]
        step_trace[-1]["inputs_count"] = len(baseline_rows) + len(recent_rows)
        step_trace[-1]["outputs_count"] = len(baseline_emb) + len(recent_emb)
        step_trace[-1]["notes"] = f"baseline={len(baseline_emb)} recent={len(recent_emb)}"

    if len(baseline_emb) < min_samples_per_window or len(recent_emb) < min_samples_per_window:
        step_trace.append({
            "step": "compute_drift",
            "status": "ok",
            "started_at": step_trace[-1]["ended_at"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "notes": f"insufficient_samples (min {min_samples_per_window} per window)",
        })
        summary = {
            "metrics": {"centroid_shift": None, "mmd": None, "ks": None, "neighbor_stability": None, "norm_drift": None},
            "drift_detected": False,
            "cause": None,
            "examples": [],
            "per_type": {},
            "recommendations": ["Collect more risk signals with embeddings in baseline and recent windows."],
            "reason": "insufficient_samples",
            "n_baseline": len(baseline_emb),
            "n_recent": len(recent_emb),
        }
        ended_at = datetime.now(timezone.utc).isoformat()
        run_id = persist_agent_run(
            supabase, household_id, "graph_drift",
            started_at=started_at, ended_at=ended_at, status="completed",
            step_trace=step_trace, summary_json=summary, dry_run=dry_run,
        )
        return {
            "step_trace": step_trace,
            "summary_json": summary,
            "status": "ok",
            "started_at": started_at,
            "ended_at": ended_at,
            "run_id": run_id,
        }

    # Step 2: compute drift metrics
    with step(ctx, step_trace, "compute_drift_metrics"):
        centroid_shift = round(_centroid_shift(recent_emb, baseline_emb), 4)
        mmd = round(compute_mmd_or_energy_distance(baseline_emb, recent_emb, use_energy=True), 4)
        ks = round(_ks_aggregate(baseline_emb, recent_emb), 4)
        norm_drift = round(_embedding_norm_drift(baseline_emb, recent_emb), 4)
        neighbor_stability_delta = round(_neighbor_stability(baseline_emb, recent_emb, k=K_NEIGHBORS), 4)
        step_trace[-1]["outputs_count"] = 5
        step_trace[-1]["notes"] = f"centroid_shift={centroid_shift} mmd={mmd} ks={ks}"

    # Step 3: root-cause
    with step(ctx, step_trace, "root_cause_analysis"):
        c_b = centroid(baseline_emb)
        examples = _prototype_incidents(recent_rows, c_b or []) if c_b else []
        model_tag_changed = False
        try:
            checkpoints = {r.get("checkpoint_id") or r.get("model_name") for r in (baseline_rows + recent_rows) if r.get("checkpoint_id") or r.get("model_name")}
            if len(checkpoints) > 1:
                model_tag_changed = True
        except Exception:
            pass
        signal_types_baseline = set()
        signal_types_recent = set()
        try:
            for r in baseline_rows:
                mid = r.get("risk_signal_id")
                if mid:
                    sig = supabase.table("risk_signals").select("signal_type").eq("id", mid).eq("household_id", household_id).limit(1).execute()
                    if sig.data and sig.data[0]:
                        signal_types_baseline.add(sig.data[0].get("signal_type", ""))
            for r in recent_rows:
                mid = r.get("risk_signal_id")
                if mid:
                    sig = supabase.table("risk_signals").select("signal_type").eq("id", mid).eq("household_id", household_id).limit(1).execute()
                    if sig.data and sig.data[0]:
                        signal_types_recent.add(sig.data[0].get("signal_type", ""))
        except Exception:
            pass
        new_pattern = bool(signal_types_recent - signal_types_baseline)
        cause = _cause_label(model_tag_changed, new_pattern)
        per_type = {}
        step_trace[-1]["notes"] = f"cause={cause} examples={len(examples)}"

    # Composite drift decision
    drift_detected = centroid_shift > drift_threshold or (mmd > 0.2 and ks > 0.3)
    metrics = {
        "centroid_shift": centroid_shift,
        "mmd": mmd,
        "ks": ks,
        "neighbor_stability": neighbor_stability_delta,
        "norm_drift": norm_drift,
    }
    recommendations = []
    if drift_detected:
        recommendations.append("Run redteam regression to validate similar incidents and watchlists.")
        recommendations.append("Consider retrain or threshold adjustment.")
        recommendations.append("Review prototype incident risk_signal_ids for patterns.")
    else:
        recommendations.append("No significant drift; continue monitoring.")

    summary = {
        "metrics": metrics,
        "drift_detected": drift_detected,
        "cause": cause,
        "examples": examples,
        "per_type": per_type,
        "recommendations": recommendations,
        "model_version_tag": model_tag,
    }

    # Step 4: emit outputs
    risk_signal_id = None
    with step(ctx, step_trace, "emit_outputs"):
        if drift_detected and not dry_run and supabase:
            severity = 3 if centroid_shift > 0.25 else 2
            explanation = {
                "summary": f"Embedding drift detected: centroid_shift={centroid_shift:.3f}, MMD={mmd:.3f}. Cause: {cause}.",
                "metrics": metrics,
                "cause": cause,
                "example_risk_signal_ids": examples,
                "model_available": True,
            }
            recommended_action = {
                "checklist": recommendations,
                "retrain_suggested": True,
                "action": "review",
            }
            risk_signal_id = upsert_risk_signal(
                supabase, household_id,
                {
                    "signal_type": "drift_warning",
                    "severity": severity,
                    "score": float(centroid_shift),
                    "explanation": explanation,
                    "recommended_action": recommended_action,
                    "status": "open",
                },
                dry_run=False,
            )
            if risk_signal_id:
                summary["risk_signal_id"] = risk_signal_id
            summary_id = upsert_summary(
                supabase, household_id,
                period_start=recent_start, period_end=recent_end,
                summary_text=f"Weekly drift check: {'Drift detected' if drift_detected else 'No significant drift'}. {cause}.",
                summary_json=summary,
                dry_run=False,
            )
            if summary_id:
                summary["summary_id"] = summary_id
        step_trace[-1]["outputs_count"] = (1 if risk_signal_id else 0) + (1 if summary.get("summary_id") else 0)
        step_trace[-1]["artifacts_refs"] = {"risk_signal_ids": [risk_signal_id] if risk_signal_id else [], "summary_ids": [summary.get("summary_id")] if summary.get("summary_id") else []}

    ended_at = datetime.now(timezone.utc).isoformat()
    run_id = persist_agent_run(
        supabase, household_id, "graph_drift",
        started_at=started_at, ended_at=ended_at, status="completed",
        step_trace=step_trace, summary_json=summary, dry_run=dry_run,
    )
    return {
        "step_trace": step_trace,
        "summary_json": summary,
        "status": "ok",
        "started_at": started_at,
        "ended_at": ended_at,
        "run_id": run_id,
    }
