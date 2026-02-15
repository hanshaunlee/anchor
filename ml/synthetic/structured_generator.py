"""
Structured synthetic heterograph generator for HGT training.

Fraud labels emerge from a structural generative process (not random):
- Two populations: Normal (Poisson degree, same-session, low device reuse, uniform time)
  vs Fraud (power-law degree, cross-session, device hubs, bursty)
- Community structure: 5–10 session communities; fraud bridges communities
- Temporal: Normal ~ Exponential(λ=1) inter-event; Fraud = bursts then silence
- Device bipartite: Normal ~1 device; Fraud = many entities share device hubs
- Labels assigned probabilistically via P(y=1)=σ(α·structural_features) after graph build
- Validation: two-sample t-tests, Cohen's d ≥ 0.5 for ≥3 metrics; separability asserts
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Strong logistic coefficients so P(fraud) is highly determined by structure (standardized features)
DEFAULT_ALPHA = {
    "cross_session_ratio": 1.8,
    "triangle_count": 1.0,
    "k_core_number": 1.0,
    "betweenness": 0.9,
    "device_reuse_count": 1.4,
    "temporal_burst_score": 1.2,
}


@dataclass
class StructuredGeneratorConfig:
    """Config for structured synthetic graph generation. Tuned for strong signal (target ROC-AUC ≥ 0.75)."""

    n_entities: int = 600
    fraud_regime_ratio: float = 0.28  # proportion of entities in "fraud" generative regime
    n_sessions: int = 180
    n_communities: int = 8
    n_devices: int = 200
    # Normal: few sessions, one community, almost no device sharing, regular timing
    normal_degree_lambda: float = 12.0
    normal_sessions_per_entity_mean: float = 4.0
    normal_device_reuse_prob: float = 0.02
    # Fraud: many sessions, multi-community, device hubs, bursty
    fraud_degree_alpha: float = 2.5
    fraud_degree_min: int = 20
    fraud_sessions_per_entity_mean: float = 38.0
    fraud_device_reuse_prob: float = 0.88
    # Events: normal regular; fraud tight bursts then long gap
    normal_inter_event_lambda: float = 2.0
    fraud_burst_size_mean: float = 14.0
    fraud_burst_gap_mean: float = 90.0
    fraud_burst_intra_scale: float = 0.08  # Exponential scale inside burst (small = very tight)
    # Explicit cross-community bridge edges for fraud (boosts cross_session_ratio, betweenness)
    fraud_bridge_edges_per_entity: int = 12
    # Validation (stricter = stronger signal required)
    min_cohen_d_metrics: int = 4
    min_cohen_d: float = 0.6
    separability_triangle_threshold: float = 1.0
    separability_cross_session_threshold: float = 0.08
    max_regenerate_attempts: int = 8
    alpha: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ALPHA))
    seed: int = 42

    def __post_init__(self) -> None:
        """Override from env for Modal/sweep runs: ANCHOR_STRUCTURED_N_ENTITIES, ANCHOR_STRUCTURED_N_SESSIONS."""
        n_ent = os.environ.get("ANCHOR_STRUCTURED_N_ENTITIES")
        if n_ent is not None:
            self.n_entities = int(n_ent)
        n_sess = os.environ.get("ANCHOR_STRUCTURED_N_SESSIONS")
        if n_sess is not None:
            self.n_sessions = int(n_sess)


def _power_law_sample(rng: np.random.Generator, alpha: float, x_min: int, size: int) -> np.ndarray:
    """Sample integers from discrete power law P(x) ∝ x^{-alpha}, x >= x_min."""
    u = rng.uniform(0, 1, size=size)
    # CDF for continuous power law: F(x) = 1 - (x_min/x)^{alpha-1}; invert
    x = x_min * (1 - u) ** (-1 / (alpha - 1))
    return np.clip(np.round(x).astype(int), x_min, None)


def generate_entities(config: StructuredGeneratorConfig, rng: np.random.Generator) -> tuple[list[dict], np.ndarray]:
    """Generate entity list and regime (0=normal, 1=fraud) per entity."""
    n_fraud_regime = int(config.n_entities * config.fraud_regime_ratio)
    n_normal = config.n_entities - n_fraud_regime
    regime = np.zeros(config.n_entities, dtype=np.int64)
    regime[:n_fraud_regime] = 1
    rng.shuffle(regime)
    entities = [{"id": f"e_{i}"} for i in range(config.n_entities)]
    return entities, regime


def generate_sessions(config: StructuredGeneratorConfig, rng: np.random.Generator) -> list[dict]:
    """Generate sessions with community assignment and started_at."""
    base_ts = 1000.0
    sessions = []
    for i in range(config.n_sessions):
        comm = i % config.n_communities
        ts = base_ts + i * 10.0 + rng.uniform(0, 5)
        sessions.append({"id": f"s_{i}", "community_id": comm, "started_at": ts})
    return sessions


def generate_devices(config: StructuredGeneratorConfig) -> list[dict]:
    """Generate device list."""
    return [{"id": f"d_{i}"} for i in range(config.n_devices)]


def assign_entities_to_devices(
    config: StructuredGeneratorConfig,
    entity_ids: list[str],
    regime: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """entity_id -> list of device_ids. Normal: 1 non-hub device; Fraud: share hub devices."""
    entity_to_devices: dict[str, list[str]] = {e: [] for e in entity_ids}
    device_ids = [f"d_{i}" for i in range(config.n_devices)]
    n_hubs = max(2, int(config.n_devices * 0.15))
    hub_ids = set(device_ids[:n_hubs])
    non_hub_ids = [d for d in device_ids if d not in hub_ids]
    normal_ids = [entity_ids[i] for i in range(len(entity_ids)) if regime[i] == 0]
    fraud_ids = [entity_ids[i] for i in range(len(entity_ids)) if regime[i] == 1]

    for e in normal_ids:
        if rng.random() < config.normal_device_reuse_prob and len(normal_ids) > 1:
            other = rng.choice(normal_ids)
            if other != e and entity_to_devices[other] and entity_to_devices[other][0] in non_hub_ids:
                entity_to_devices[e] = [rng.choice(entity_to_devices[other])]
            else:
                entity_to_devices[e] = [rng.choice(non_hub_ids)]
        else:
            entity_to_devices[e] = [rng.choice(non_hub_ids)]

    for e in fraud_ids:
        if rng.random() < config.fraud_device_reuse_prob:
            n_hub_use = min(rng.integers(2, 5), len(hub_ids))
            entity_to_devices[e] = list(rng.choice(list(hub_ids), size=n_hub_use, replace=False))
        else:
            entity_to_devices[e] = [rng.choice(list(hub_ids))]
    return entity_to_devices


def assign_entities_to_sessions(
    config: StructuredGeneratorConfig,
    entity_ids: list[str],
    regime: np.ndarray,
    sessions: list[dict],
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    """entity_id -> list of session indices. Normal: one community only; Fraud: spread across ≥3 communities."""
    entity_to_sessions: dict[str, list[int]] = {e: [] for e in entity_ids}
    session_communities = np.array([s["community_id"] for s in sessions])
    n_sess = len(sessions)
    sessions_by_comm: dict[int, list[int]] = {}
    for j in range(n_sess):
        c = session_communities[j]
        sessions_by_comm.setdefault(c, []).append(j)

    for i, e in enumerate(entity_ids):
        if regime[i] == 0:
            n_s = max(1, min(int(rng.poisson(config.normal_sessions_per_entity_mean)), 10))
            comm = rng.integers(0, config.n_communities)
            cand = sessions_by_comm.get(comm, list(range(n_sess)))
            if len(cand) < n_s:
                cand = cand * (n_s // len(cand) + 1)
            chosen = rng.choice(cand, size=min(n_s, len(cand)), replace=False)
            entity_to_sessions[e] = np.unique(chosen).tolist()
        else:
            n_s = max(10, int(rng.poisson(config.fraud_sessions_per_entity_mean)))
            n_s = min(n_s, n_sess)
            n_comms_use = 3 + int(rng.integers(0, max(1, config.n_communities - 2)))
            n_comms_use = min(n_comms_use, config.n_communities)
            comms_to_use = rng.choice(config.n_communities, size=n_comms_use, replace=False)
            cand = []
            for c in comms_to_use:
                cand.extend(sessions_by_comm.get(int(c), []))
            cand = list(set(cand))
            if len(cand) < n_s:
                cand = list(range(n_sess))
            chosen = rng.choice(cand, size=min(n_s, len(cand)), replace=False)
            entity_to_sessions[e] = np.unique(chosen).tolist()
    return entity_to_sessions


def generate_events_and_mentions(
    config: StructuredGeneratorConfig,
    entity_ids: list[str],
    regime: np.ndarray,
    sessions: list[dict],
    entity_to_sessions: dict[str, list[int]],
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    """Generate utterances (events) and mentions; return also per-entity inter-event deltas for burstiness."""
    utterances: list[dict] = []
    mentions: list[dict] = []
    entity_inter_times: dict[str, list[float]] = {e: [] for e in entity_ids}
    base_ts = 1000.0
    u_id = 0
    intents = ("call", "reminder", "transfer", "balance", "pay_bill", "alerts")

    for sess_idx, sess in enumerate(sessions):
        sess_id = sess["id"]
        sess_ts = sess["started_at"]
        entities_in_sess = [e for e in entity_ids if sess_idx in entity_to_sessions[e]]
        if not entities_in_sess:
            continue
        # Order entities by regime so we can simulate burstiness for fraud
        fraud_in_sess = [e for e in entities_in_sess if regime[entity_ids.index(e)] == 1]
        normal_in_sess = [e for e in entities_in_sess if regime[entity_ids.index(e)] == 0]
        t = float(sess_ts)
        events_this_session: list[tuple[float, list[str]]] = []

        # Normal: regular spacing, fewer events per session
        n_normal_events = max(1, len(normal_in_sess) * 2)
        for _ in range(n_normal_events):
            t += rng.exponential(config.normal_inter_event_lambda)
            n_ent = min(rng.integers(1, 3), len(entities_in_sess))
            chosen = rng.choice(entities_in_sess, size=n_ent, replace=False)
            events_this_session.append((t, list(chosen)))

        # Fraud: tight bursts (small intra-burst delta) then long gap
        for _ in range(len(fraud_in_sess) * 2):
            burst_size = max(3, int(rng.poisson(config.fraud_burst_size_mean)))
            for __ in range(burst_size):
                t += 0.02 + rng.exponential(config.fraud_burst_intra_scale)
                n_ent = min(rng.integers(2, 5), len(entities_in_sess))
                chosen = rng.choice(entities_in_sess, size=n_ent, replace=False)
                events_this_session.append((t, list(chosen)))
            t += rng.exponential(config.fraud_burst_gap_mean)

        events_this_session.sort(key=lambda x: x[0])
        for t, ents in events_this_session:
            uid = f"u_{u_id}"
            u_id += 1
            utterances.append({
                "id": uid,
                "session_id": sess_id,
                "intent": rng.choice(intents),
                "ts": t,
            })
            for e in ents:
                mentions.append({"utterance_id": uid, "entity_id": e})
                entity_inter_times[e].append(t)

    # Compute inter-event deltas per entity (for burstiness)
    for e in entity_ids:
        times = sorted(entity_inter_times[e])
        if len(times) >= 2:
            entity_inter_times[e] = np.diff(times).tolist()
        else:
            entity_inter_times[e] = []

    return utterances, mentions, entity_inter_times


def build_co_occurs_and_primary_community(
    sessions: list[dict],
    entity_ids: list[str],
    entity_to_sessions: dict[str, list[int]],
    mentions: list[dict],
) -> tuple[list[dict], np.ndarray, dict[str, int]]:
    """Build CO_OCCURS from same-session co-occurrence. Return relationships, session_community per session, entity primary community."""
    session_communities = np.array([s["community_id"] for s in sessions])
    entity_sessions: dict[str, list[int]] = entity_to_sessions
    # Primary community = most frequent session community for this entity
    primary_comm: dict[str, int] = {}
    for e in entity_ids:
        sess_list = entity_sessions.get(e, [])
        if not sess_list:
            primary_comm[e] = 0
            continue
        comms = session_communities[sess_list]
        primary_comm[e] = int(np.bincount(comms).argmax())

    seen: set[tuple[str, str]] = set()
    relationships: list[dict] = []
    t0 = 1000.0
    for sess_idx, sess in enumerate(sessions):
        comm = sess["community_id"]
        entities_in_sess = [e for e in entity_ids if sess_idx in entity_sessions.get(e, [])]
        for i in range(len(entities_in_sess)):
            for j in range(i + 1, len(entities_in_sess)):
                a, b = entities_in_sess[i], entities_in_sess[j]
                if a > b:
                    a, b = b, a
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                ts_mid = t0 + sess_idx * 10
                relationships.append({
                    "src_entity_id": a,
                    "dst_entity_id": b,
                    "rel_type": "CO_OCCURS",
                    "first_seen_at": ts_mid - 1,
                    "last_seen_at": ts_mid,
                    "count": 1,
                    "weight": 1.0,
                })
    return relationships, session_communities, primary_comm


def add_cross_community_bridge_edges(
    relationships: list[dict],
    entity_ids: list[str],
    regime: np.ndarray,
    primary_comm: dict[str, int],
    sessions: list[dict],
    config: StructuredGeneratorConfig,
    rng: np.random.Generator,
    t0: float = 1000.0,
) -> None:
    """Add CO_OCCURS edges between fraud entities and entities in other communities (in-place). Boosts cross_session_ratio and betweenness."""
    session_communities = [s["community_id"] for s in sessions]
    entities_by_comm: dict[int, list[str]] = {}
    for e in entity_ids:
        c = primary_comm.get(e, 0)
        entities_by_comm.setdefault(c, []).append(e)
    fraud_ids = [entity_ids[i] for i in range(len(entity_ids)) if regime[i] == 1]
    seen = set()
    for r in relationships:
        a, b = r["src_entity_id"], r["dst_entity_id"]
        if a > b:
            a, b = b, a
        seen.add((a, b))
    n_added = 0
    k = config.fraud_bridge_edges_per_entity
    for e in fraud_ids:
        my_comm = primary_comm.get(e, 0)
        other_comms = [c for c in entities_by_comm if c != my_comm]
        if not other_comms:
            continue
        candidates = []
        for c in other_comms:
            candidates.extend(entities_by_comm[c])
        if len(candidates) < k:
            continue
        chosen = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        for u in chosen:
            a, b = (e, u) if e < u else (u, e)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            relationships.append({
                "src_entity_id": a,
                "dst_entity_id": b,
                "rel_type": "CO_OCCURS",
                "first_seen_at": t0,
                "last_seen_at": t0 + 1,
                "count": 1,
                "weight": 1.0,
            })
            n_added += 1
    if n_added > 0:
        logger.debug("Added %d cross-community bridge edges for fraud entities", n_added)


def compute_structural_features(
    entity_ids: list[str],
    relationships: list[dict],
    entity_to_devices: dict[str, list[str]],
    entity_inter_times: dict[str, list[float]],
    primary_comm: dict[str, int],
    session_communities: np.ndarray,
    entity_to_sessions: dict[str, list[int]],
) -> np.ndarray:
    """Compute (n_entities, n_features): cross_session_ratio, triangle_count, k_core_number, betweenness, device_reuse_count, temporal_burst_score."""
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available; using placeholder structural features")
        n = len(entity_ids)
        return np.zeros((n, 6))

    G = nx.Graph()
    G.add_nodes_from(entity_ids)
    for r in relationships:
        if r.get("rel_type") != "CO_OCCURS":
            continue
        a, b = r.get("src_entity_id"), r.get("dst_entity_id")
        if a and b and a in entity_ids and b in entity_ids:
            G.add_edge(a, b)

    e2i = {e: i for i, e in enumerate(entity_ids)}
    n = len(entity_ids)
    features = np.zeros((n, 6))

    # cross_session_ratio: fraction of edges to entities in a different primary community
    for e in entity_ids:
        i = e2i[e]
        neighbors = list(G.neighbors(e))
        if not neighbors:
            features[i, 0] = 0.0
            continue
        other_comm = sum(1 for u in neighbors if primary_comm.get(u, 0) != primary_comm.get(e, 0))
        features[i, 0] = other_comm / len(neighbors)

    # triangle_count, k_core, betweenness
    triangles = nx.triangles(G)
    for e in entity_ids:
        i = e2i[e]
        features[i, 1] = triangles.get(e, 0)
    try:
        k_core = nx.core_number(G)
        for e in entity_ids:
            features[e2i[e], 2] = k_core.get(e, 0)
    except Exception:
        pass
    try:
        between = nx.betweenness_centrality(G)
        for e in entity_ids:
            features[e2i[e], 3] = between.get(e, 0.0)
    except Exception:
        pass

    # device_reuse_count: number of other entities sharing at least one device with this entity
    device_to_entities: dict[str, set[str]] = {}
    for e, devs in entity_to_devices.items():
        for d in devs:
            device_to_entities.setdefault(d, set()).add(e)
    for e in entity_ids:
        i = e2i[e]
        devs = entity_to_devices.get(e, [])
        others = set()
        for d in devs:
            others.update(device_to_entities.get(d, set()))
        others.discard(e)
        features[i, 4] = len(others)

    # temporal_burst_score B = (σ(Δt) - μ(Δt)) / (σ(Δt) + μ(Δt)); high when bursty
    for e in entity_ids:
        i = e2i[e]
        deltas = entity_inter_times.get(e, [])
        if len(deltas) < 2:
            features[i, 5] = 0.0
            continue
        arr = np.array(deltas, dtype=float)
        mu, sig = arr.mean(), arr.std()
        if sig + mu <= 0:
            features[i, 5] = 0.0
        else:
            features[i, 5] = (sig - mu) / (sig + mu)
        features[i, 5] = np.clip(features[i, 5], -1, 1)

    return features


def assign_labels_probabilistically(
    features: np.ndarray,
    alpha: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """P(y=1) = sigmoid(α·x); standardize features then sample. No leakage."""
    order = ["cross_session_ratio", "triangle_count", "k_core_number", "betweenness", "device_reuse_count", "temporal_burst_score"]
    x = features.copy()
    for c in range(x.shape[1]):
        col = x[:, c]
        if col.std() > 1e-9:
            x[:, c] = (col - col.mean()) / col.std()
    coef = np.array([alpha.get(name, 0) for name in order], dtype=float)
    logit = x @ coef
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    y = (rng.random(len(prob)) < prob).astype(np.int64)
    return y


def validate_and_log(
    entity_ids: list[str],
    regime: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    relationships: list[dict],
    config: StructuredGeneratorConfig,
) -> dict[str, Any]:
    """Compute stats, run t-tests and Cohen's d, assert separability. Return stats dict."""
    from scipy import stats

    normal_mask = regime == 0
    fraud_mask = regime == 1
    label_fraud = labels == 1
    label_normal = labels == 0
    n_fraud = int(label_fraud.sum())
    n_normal = int(label_normal.sum())

    # Build degree from relationships
    degree = np.zeros(len(entity_ids))
    e2i = {e: i for i, e in enumerate(entity_ids)}
    for r in relationships:
        if r.get("rel_type") != "CO_OCCURS":
            continue
        a, b = r.get("src_entity_id"), r.get("dst_entity_id")
        if a in e2i:
            degree[e2i[a]] += 1
        if b in e2i:
            degree[e2i[b]] += 1

    feature_names = ["cross_session_ratio", "triangle_count", "k_core_number", "betweenness", "device_reuse_count", "temporal_burst_score"]
    # Degree distribution: two-sample t-test and Kolmogorov-Smirnov
    degree_fraud = degree[label_fraud]
    degree_normal = degree[label_normal]
    degree_ttest_pvalue = None
    degree_cohen_d = None
    degree_ks_statistic = None
    degree_ks_pvalue = None
    if degree_fraud.size >= 2 and degree_normal.size >= 2:
        t_deg, degree_ttest_pvalue = stats.ttest_ind(degree_fraud, degree_normal)
        pooled = np.sqrt((degree_fraud.var() + degree_normal.var()) / 2)
        degree_cohen_d = float(abs((degree_fraud.mean() - degree_normal.mean()) / pooled)) if pooled > 1e-9 else 0.0
        ks_stat, degree_ks_pvalue = stats.ks_2samp(degree_fraud, degree_normal)
        degree_ks_statistic = float(ks_stat)

    stats_dict: dict[str, Any] = {
        "label_prevalence": {"n_positive": int(n_fraud), "n_negative": int(n_normal), "n": len(labels)},
        "mean_degree_fraud": float(degree[label_fraud].mean()) if n_fraud else 0,
        "mean_degree_normal": float(degree[label_normal].mean()) if n_normal else 0,
        "degree_ttest_pvalue": degree_ttest_pvalue,
        "degree_cohen_d": degree_cohen_d,
        "degree_ks_statistic": degree_ks_statistic,
        "degree_ks_pvalue": degree_ks_pvalue,
        "mean_triangle_fraud": None,
        "mean_triangle_normal": None,
        "mean_cross_session_ratio_fraud": None,
        "mean_cross_session_ratio_normal": None,
        "cohen_d": {},
        "t_test_pvalue": {},
        "modularity": None,
        "k_core_dist_fraud": None,
        "k_core_dist_normal": None,
    }

    cohen_d_count = 0
    for idx, name in enumerate(feature_names):
        x_f = features[label_fraud, idx]
        x_n = features[label_normal, idx]
        if x_f.size < 2 or x_n.size < 2:
            continue
        t, p = stats.ttest_ind(x_f, x_n)
        pooled_std = np.sqrt((x_f.var() + x_n.var()) / 2)
        if pooled_std > 1e-9:
            d = abs(float((x_f.mean() - x_n.mean()) / pooled_std))
        else:
            d = 0.0
        stats_dict["cohen_d"][name] = d
        stats_dict["t_test_pvalue"][name] = float(p)
        if d >= config.min_cohen_d:
            cohen_d_count += 1
        if name == "triangle_count":
            stats_dict["mean_triangle_fraud"] = float(x_f.mean())
            stats_dict["mean_triangle_normal"] = float(x_n.mean())
        if name == "cross_session_ratio":
            stats_dict["mean_cross_session_ratio_fraud"] = float(x_f.mean())
            stats_dict["mean_cross_session_ratio_normal"] = float(x_n.mean())

    # Modularity and k-core distribution (build G from relationships)
    try:
        import networkx as nx
        from networkx.algorithms import community
        G = nx.Graph()
        G.add_nodes_from(entity_ids)
        for r in relationships:
            if r.get("rel_type") != "CO_OCCURS":
                continue
            a, b = r.get("src_entity_id"), r.get("dst_entity_id")
            if a and b:
                G.add_edge(a, b)
        comp = community.greedy_modularity_communities(G)
        mod = community.modularity(G, comp)
        stats_dict["modularity"] = float(mod)
        k_core = nx.core_number(G)
        k_f = [k_core.get(e, 0) for e in entity_ids if labels[e2i[e]] == 1]
        k_n = [k_core.get(e, 0) for e in entity_ids if labels[e2i[e]] == 0]
        stats_dict["k_core_dist_fraud"] = {"mean": float(np.mean(k_f)), "max": int(max(k_f))} if k_f else None
        stats_dict["k_core_dist_normal"] = {"mean": float(np.mean(k_n)), "max": int(max(k_n))} if k_n else None
    except Exception:
        pass

    if stats_dict["mean_triangle_fraud"] is not None and stats_dict["mean_triangle_normal"] is not None:
        assert abs(stats_dict["mean_triangle_fraud"] - stats_dict["mean_triangle_normal"]) >= config.separability_triangle_threshold, (
            "Triangle separability failed: abs(mean_triangle_fraud - mean_triangle_normal) < threshold"
        )
    if stats_dict["mean_cross_session_ratio_fraud"] is not None and stats_dict["mean_cross_session_ratio_normal"] is not None:
        assert abs(stats_dict["mean_cross_session_ratio_fraud"] - stats_dict["mean_cross_session_ratio_normal"]) >= config.separability_cross_session_threshold, (
            "Cross-session ratio separability failed"
        )
    assert cohen_d_count >= config.min_cohen_d_metrics, (
        f"Need at least {config.min_cohen_d_metrics} metrics with Cohen's d >= {config.min_cohen_d}, got {cohen_d_count}"
    )

    logger.info(
        "Structured synthetic stats: mean_degree fraud=%.2f normal=%.2f, mean_triangle fraud=%.2f normal=%.2f, "
        "mean_cross_session_ratio fraud=%.3f normal=%.3f, Cohen d >= %.2f for %d metrics",
        stats_dict["mean_degree_fraud"], stats_dict["mean_degree_normal"],
        stats_dict["mean_triangle_fraud"] or 0, stats_dict["mean_triangle_normal"] or 0,
        stats_dict["mean_cross_session_ratio_fraud"] or 0, stats_dict["mean_cross_session_ratio_normal"] or 0,
        config.min_cohen_d, cohen_d_count,
    )
    return stats_dict


def generate_structured_heterograph(
    config: StructuredGeneratorConfig | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], np.ndarray, dict[str, Any]]:
    """
    Generate sessions, utterances, entities, mentions, relationships, devices, and entity labels.
    Labels are assigned probabilistically from structural features (no leakage).
    Validates separability and Cohen's d; raises if not met (up to max_regenerate_attempts).
    Returns: (sessions, utterances, entities, mentions, relationships, devices, entity_labels, stats_dict).
    """
    config = config or StructuredGeneratorConfig()
    rng = np.random.default_rng(config.seed)

    for attempt in range(config.max_regenerate_attempts):
        try:
            entities, regime = generate_entities(config, rng)
            entity_ids = [e["id"] for e in entities]
            sessions = generate_sessions(config, rng)
            devices = generate_devices(config)
            entity_to_devices = assign_entities_to_devices(config, entity_ids, regime, rng)
            entity_to_sessions = assign_entities_to_sessions(config, entity_ids, regime, sessions, rng)
            utterances, mentions, entity_inter_times = generate_events_and_mentions(
                config, entity_ids, regime, sessions, entity_to_sessions, rng
            )
            relationships, session_communities, primary_comm = build_co_occurs_and_primary_community(
                sessions, entity_ids, entity_to_sessions, mentions
            )
            add_cross_community_bridge_edges(
                relationships, entity_ids, regime, primary_comm, sessions, config, rng
            )
            # Ensure we have enough edges for graph metrics
            if len(relationships) < 50:
                raise ValueError("Too few CO_OCCURS edges")
            features = compute_structural_features(
                entity_ids, relationships, entity_to_devices, entity_inter_times,
                primary_comm, session_communities, entity_to_sessions,
            )
            labels = assign_labels_probabilistically(features, config.alpha, rng)
            stats_dict = validate_and_log(
                entity_ids, regime, labels, features, relationships, config
            )
            # Build device list for builder (we have entity_to_devices; devices already created)
            device_list = [{"id": f"d_{i}"} for i in range(config.n_devices)]
            return sessions, utterances, entities, mentions, relationships, device_list, labels, stats_dict
        except (AssertionError, ValueError) as e:
            logger.warning("Structured generation attempt %d failed: %s", attempt + 1, e)
            if attempt == config.max_regenerate_attempts - 1:
                raise
            rng = np.random.default_rng(config.seed + attempt + 1)
    raise RuntimeError("Structured generation failed after max attempts")
