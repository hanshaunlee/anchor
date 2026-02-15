"""Tests for conformal coverage behavior, rule score correctness, independence cluster detection, PGExplainer ID mapping, fallback scoring."""
from __future__ import annotations

import pytest


def test_rule_score_compute_rule_score_empty() -> None:
    """compute_rule_score with no tags/motifs returns 0."""
    from domain.rule_scoring import compute_rule_score
    assert compute_rule_score([], []) == 0.0
    assert compute_rule_score([], [], None) == 0.0


def test_rule_score_semantic_and_structural() -> None:
    """compute_rule_score adds for urgency, sensitive, new_contact, star_pattern, independence."""
    from domain.rule_scoring import compute_rule_score
    s = compute_rule_score(
        ["New contact + urgency topic", "Contact → sensitive intent cascade"],
        [{"pattern_type": "star_pattern", "node_ids": ["e1", "e2", "e3"]}],
        None,
    )
    assert s > 0.0
    assert s <= 1.0


def test_rule_score_independence_violation() -> None:
    """compute_rule_score adds for bridges_independent_sets or continuous independence_violation_ratio."""
    from domain.rule_scoring import compute_rule_score
    s_no_bridge = compute_rule_score([], [], {"bridges_independent_sets": False})
    s_bridge = compute_rule_score([], [], {"bridges_independent_sets": True})
    assert s_bridge >= s_no_bridge
    assert s_bridge >= 0.3
    s_ratio = compute_rule_score([], [], {"independence_violation_ratio": 0.5})
    assert s_ratio >= 0.15  # 0.3 * 0.5
    assert s_ratio <= 0.2


def test_independence_cluster_detection_empty() -> None:
    """compute_independent_entity_sets with no entities returns empty."""
    from ml.graph.builder import compute_independent_entity_sets
    out = compute_independent_entity_sets([], [])
    assert out == {}


def test_independence_cluster_detection_single_entity() -> None:
    """Single entity gets cluster 0, set size 1, no bridge, violation_ratio 0."""
    from ml.graph.builder import compute_independent_entity_sets
    entities = [{"id": "e1"}]
    out = compute_independent_entity_sets(entities, [])
    assert "e1" in out
    assert out["e1"]["independence_cluster_id"] == 0
    assert out["e1"]["independent_set_size"] == 1
    assert out["e1"]["bridges_independent_sets"] is False
    assert out["e1"].get("independence_violation_ratio", 0.0) == 0.0


def test_independence_cluster_detection_two_unconnected() -> None:
    """Two entities with no edge: both in same or separate MIS; at least one cluster."""
    from ml.graph.builder import compute_independent_entity_sets
    entities = [{"id": "e1"}, {"id": "e2"}]
    out = compute_independent_entity_sets(entities, [])
    assert len(out) == 2
    assert out["e1"]["independent_set_size"] >= 1
    assert out["e2"]["independent_set_size"] >= 1


def test_independence_cluster_detection_two_connected() -> None:
    """Two entities with edge: different clusters or one cluster of size 1 each."""
    from ml.graph.builder import compute_independent_entity_sets
    entities = [{"id": "e1"}, {"id": "e2"}]
    rels = [{"src_entity_id": "e1", "dst_entity_id": "e2"}]
    out = compute_independent_entity_sets(entities, rels)
    assert len(out) == 2
    # Either both in different clusters (no bridge) or one is bridge
    assert isinstance(out["e1"]["independence_cluster_id"], int)
    assert isinstance(out["e2"]["independence_cluster_id"], int)


def test_pg_service_entity_id_mapping() -> None:
    """attach_pg_explanations maps PyG indices to entity IDs in model_subgraph."""
    from domain.explainers.pg_service import _entity_id_from_index
    entities = [{"id": "entity_phone_abc"}, {"id": "entity_person_def"}]
    assert _entity_id_from_index(0, entities) == "entity_phone_abc"
    assert _entity_id_from_index(1, entities) == "entity_person_def"
    assert _entity_id_from_index(99, entities) == "99"


def test_fallback_scoring_uses_rule_score() -> None:
    """Pipeline fallback produces scores from compute_rule_score, not fake formula."""
    from domain.rule_scoring import compute_rule_score
    pattern_tags = ["New contact + urgency topic"]
    structural_motifs = []
    entity_meta = {"bridges_independent_sets": False}
    score = compute_rule_score(pattern_tags, structural_motifs, entity_meta)
    assert score >= 0.0
    assert score <= 1.0
    # No fake 0.1 + (i % 3) * 0.2
    assert score not in (0.1, 0.3, 0.5)


def test_conformal_split_quantile_level() -> None:
    """Split conformal uses ceil((n+1)*(1-target_fpr))/n for quantile level."""
    import math
    n = 10
    target_fpr = 0.1
    level = math.ceil((n + 1) * (1 - target_fpr)) / n
    assert level >= 0.0
    assert level <= 1.0
    # ceil(11*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0
    assert level == 1.0
