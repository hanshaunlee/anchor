"""
Rule-based risk score from pattern tags and structural motifs.
Used when GNN is unavailable (fallback) and for model-rule fusion.
No fake placeholder scores; scores are grounded in semantic + structural signals.
Independence violation uses continuous ratio (cross_cluster_edges/degree) when available.
"""
from __future__ import annotations


def compute_rule_score(
    pattern_tags: list[str],
    structural_motifs: list[dict],
    entity_meta: dict | None = None,
) -> float:
    """
    Compute rule-only score in [0, 1] from semantic pattern tags and structural motifs.
    entity_meta optional: {
        "bridges_independent_sets": bool,
        "independence_violation_ratio": float in [0,1],  # cross_cluster_edges / degree
    }. When independence_violation_ratio is present, use it for a continuous contribution
    (up to 0.3); otherwise fall back to binary bridges_independent_sets (+0.3).
    """
    score = 0.0
    tags_lower = " ".join(pattern_tags).lower()
    motifs_lower = " ".join(str(m.get("pattern_type", "")) for m in structural_motifs).lower()

    # Semantic pattern contributions
    if "urgency" in tags_lower or "medicare" in tags_lower or "irs" in tags_lower:
        score += 0.2
    if "sensitive" in tags_lower or "share" in tags_lower or "pay" in tags_lower:
        score += 0.2
    if "new contact" in tags_lower or "new_contact" in motifs_lower:
        score += 0.2
    if "bursty" in tags_lower or "repeated contact" in tags_lower:
        score += 0.15
    if "cascade" in tags_lower or "device switching" in tags_lower:
        score += 0.15

    # Structural motif contributions
    if "star_pattern" in motifs_lower or "star" in motifs_lower:
        score += 0.3
    if "triadic" in motifs_lower or "triadic_closure" in motifs_lower:
        score += 0.2
    if "2hop_chain" in motifs_lower or "chain" in motifs_lower:
        score += 0.2

    # Independence violation: continuous ratio (cross_cluster_edges/degree) or binary bridge
    if entity_meta:
        ratio = entity_meta.get("independence_violation_ratio")
        if ratio is not None and isinstance(ratio, (int, float)):
            score += 0.3 * min(1.0, max(0.0, float(ratio)))
        elif entity_meta.get("bridges_independent_sets"):
            score += 0.3

    return min(1.0, round(score, 4))
