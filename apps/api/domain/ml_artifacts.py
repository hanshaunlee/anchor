"""
Shared ML utilities for agents: checkpoint loading, embedding fetch, drift/clustering.
Used by Graph Drift, Ring Discovery, Calibration, and Red-Team agents.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from datetime import datetime, timezone, timedelta


def load_checkpoint_or_none(checkpoint_path: str | Path | None = None) -> Any | None:
    """Load GNN checkpoint if path exists; return None otherwise (no fake model)."""
    if checkpoint_path is None:
        try:
            from config.settings import get_ml_settings
            checkpoint_path = get_ml_settings().checkpoint_path
        except Exception:
            return None
    path = Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
    if not path.is_file():
        return None
    try:
        import torch
        from ml.inference import load_model
        device = torch.device("cpu")
        model, _ = load_model(path, device)
        return model
    except Exception:
        return None


def _l2_norm(vec: list[float]) -> float:
    s = math.sqrt(sum(x * x for x in vec)) or 1e-12
    return s


def normalize(vec: list[float]) -> list[float]:
    """L2-normalize vector in-place style; returns new list."""
    n = _l2_norm(vec)
    return [x / n for x in vec]


def cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (_l2_norm(a) * _l2_norm(b))


def fetch_embeddings_window(
    supabase: Any,
    household_id: str,
    time_range: tuple[str, str],
    *,
    require_has_embedding: bool = True,
    embedding_space: str = "risk_signal",
    extra_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch risk_signal_embeddings for household in [start_iso, end_iso].
    Returns list of {embedding, risk_signal_id, created_at, severity?, signal_type?, ...}.
    """
    if embedding_space != "risk_signal":
        # Future: entity/session embeddings table
        return []
    start_iso, end_iso = time_range
    cols = ["risk_signal_id", "household_id", "embedding", "created_at", "has_embedding"]
    if extra_columns:
        cols = list(set(cols) | set(extra_columns))
    try:
        q = (
            supabase.table("risk_signal_embeddings")
            .select(",".join(cols))
            .eq("household_id", household_id)
            .gte("created_at", start_iso)
            .lte("created_at", end_iso)
        )
        if require_has_embedding:
            q = q.eq("has_embedding", True)
        r = q.execute()
        rows = r.data or []
    except Exception:
        return []
    out = []
    for row in rows:
        emb = row.get("embedding")
        if not emb or not isinstance(emb, list) or len(emb) == 0:
            continue
        if require_has_embedding and row.get("has_embedding") is False:
            continue
        out.append(dict(row))
    return out


def centroid(embeddings: list[list[float]]) -> list[float] | None:
    if not embeddings:
        return None
    dim = len(embeddings[0])
    for e in embeddings:
        if len(e) != dim:
            return None
    return [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]


def cluster_embeddings(
    embeddings: list[list[float]],
    n_clusters: int | None = None,
    min_cluster_size: int = 2,
) -> list[int]:
    """
    Cluster embeddings. Prefer HDBSCAN if available (sklearn); else KMeans.
    Returns cluster label per index (same order as embeddings). -1 = noise for HDBSCAN.
    """
    if not embeddings or len(embeddings) < 2:
        return [0] * len(embeddings) if embeddings else []
    try:
        import numpy as np
        X = np.array(embeddings, dtype=float)
    except ImportError:
        return [0] * len(embeddings)
    n = len(embeddings)
    k = n_clusters or max(2, min(10, n // 3))

    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)
        labels = clusterer.fit_predict(X)
        return labels.tolist()
    except ImportError:
        pass
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        return labels.tolist()
    except ImportError:
        pass
    return [0] * len(embeddings)


def compute_mmd_or_energy_distance(
    baseline: list[list[float]],
    recent: list[list[float]],
    *,
    use_energy: bool = True,
) -> float:
    """
    Two-sample distance. Prefer energy distance (E-statistic); fallback simple MMD-style.
    Returns non-negative value; larger = more drift.
    """
    if not baseline or not recent:
        return 0.0
    try:
        import numpy as np
        A = np.array(baseline, dtype=float)
        B = np.array(recent, dtype=float)
    except ImportError:
        # Pure Python fallback: mean pairwise distance recent vs baseline
        def mean_dist(X: list[list[float]], Y: list[list[float]]) -> float:
            total = 0.0
            count = 0
            for x in X:
                for y in Y:
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
                    total += d
                    count += 1
            return total / count if count else 0.0
        return float(mean_dist(recent, recent) + mean_dist(baseline, baseline) - 2 * mean_dist(recent, baseline))

    # Energy distance: E|X-X'| + E|Y-Y'| - 2*E|X-Y|
    def pairwise_norm(X: Any, Y: Any) -> float:
        d = X[:, None, :] - Y[None, :, :]
        return float(np.sqrt((d * d).sum(axis=2)).mean())
    xx = pairwise_norm(A, A)
    yy = pairwise_norm(B, B)
    xy = pairwise_norm(A, B)
    return max(0.0, xx + yy - 2 * xy)


def compute_mmd_rbf(
    baseline: list[list[float]],
    recent: list[list[float]],
    *,
    bandwidth: float | None = None,
) -> float:
    """
    MMD with RBF kernel. K(x,y) = exp(-||x-y||^2 / (2*bandwidth^2)).
    Bandwidth selection: if None, use median heuristic (median pairwise distance in combined sample).
    Returns non-negative MMD^2 estimate; larger = more drift.
    """
    if not baseline or not recent:
        return 0.0
    try:
        import numpy as np
        A = np.array(baseline, dtype=float)
        B = np.array(recent, dtype=float)
    except ImportError:
        return 0.0
    n_a, n_b = A.shape[0], B.shape[0]
    if bandwidth is None:
        combined = np.vstack([A, B])
        # Median heuristic: median pairwise distance (upper triangle of distance matrix)
        diff = combined[:, None, :] - combined[None, :, :]
        d = np.sqrt((diff * diff).sum(axis=2))
        triu = np.triu_indices(d.shape[0], 1)
        dists = d[triu]
        bandwidth = float(np.median(dists)) if len(dists) > 0 else 1.0
        if bandwidth <= 0:
            bandwidth = 1.0
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def rbf(X: Any, Y: Any) -> float:
        d2 = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        return float(np.exp(-gamma * d2).mean())
    k_aa = rbf(A, A)
    k_bb = rbf(B, B)
    k_ab = rbf(A, B)
    mmd2 = max(0.0, k_aa + k_bb - 2 * k_ab)
    return float(mmd2)


def compute_drift_confidence_interval(
    baseline: list[list[float]],
    recent: list[list[float]],
    metric_fn: Any,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a drift metric (e.g. centroid_shift or MMD).
    Returns (point_estimate, lower, upper).
    """
    if not baseline or not recent:
        return 0.0, 0.0, 0.0
    try:
        import numpy as np
    except ImportError:
        est = metric_fn(baseline, recent)
        return est, est, est
    n_b, n_r = len(baseline), len(recent)
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(n_bootstrap):
        idx_b = rng.integers(0, n_b, size=n_b)
        idx_r = rng.integers(0, n_r, size=n_r)
        boot_b = [baseline[i] for i in idx_b]
        boot_r = [recent[i] for i in idx_r]
        estimates.append(metric_fn(boot_b, boot_r))
    estimates_arr = np.array(estimates)
    point = float(metric_fn(baseline, recent))
    alpha = 1 - confidence
    lower = float(np.percentile(estimates_arr, 100 * alpha / 2))
    upper = float(np.percentile(estimates_arr, 100 * (1 - alpha / 2)))
    return point, lower, upper
