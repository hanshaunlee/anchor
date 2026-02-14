"""Tests for ml.graph.subgraph: extract_k_hop, get_touched_entities."""
import pytest

pytest.importorskip("torch")
import torch
from torch_geometric.data import HeteroData

from ml.graph.subgraph import extract_k_hop, get_touched_entities


def _make_small_hetero() -> HeteroData:
    data = HeteroData()
    data["entity"].x = torch.randn(5, 8)
    data["entity"].num_nodes = 5
    # edges: 0-1, 1-2, 2-3, 3-4 (chain) and 0-4
    data["entity", "co_occurs", "entity"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long
    )
    data["entity", "co_occurs", "entity"].edge_attr = torch.zeros(5, 4)
    return data


def test_extract_k_hop_seeds_only() -> None:
    data = _make_small_hetero()
    sub, g2l = extract_k_hop(data, {"entity": [1]}, k=0)
    assert sub["entity"].num_nodes == 1
    assert list(g2l["entity"].keys()) == [1]


def test_extract_k_hop_one_hop() -> None:
    data = _make_small_hetero()
    sub, g2l = extract_k_hop(data, {"entity": [0]}, k=1)
    # From 0: neighbors 1 and 4
    assert sub["entity"].num_nodes >= 2
    assert 0 in g2l["entity"] and 1 in g2l["entity"]


def test_extract_k_hop_two_hops() -> None:
    data = _make_small_hetero()
    sub, g2l = extract_k_hop(data, {"entity": [0]}, k=2)
    # 0 -> 1,4 -> 2,3
    assert sub["entity"].num_nodes >= 3


def test_get_touched_entities_from_mentions() -> None:
    entity_id_to_index = {"e1": 0, "e2": 1, "e3": 2}
    new_mentions = [{"entity_id": "e1"}, {"entity_id": "e3"}]
    touched = get_touched_entities(new_mentions, [], entity_id_to_index)
    assert set(touched) == {0, 2}


def test_get_touched_entities_from_relationships() -> None:
    entity_id_to_index = {"e1": 0, "e2": 1}
    new_relationships = [
        {"src_entity_id": "e1", "dst_entity_id": "e2"},
    ]
    touched = get_touched_entities([], new_relationships, entity_id_to_index)
    assert set(touched) == {0, 1}


def test_get_touched_entities_combined() -> None:
    entity_id_to_index = {"e1": 0, "e2": 1}
    new_mentions = [{"entity_id": "e1"}]
    new_relationships = [{"src_entity_id": "e2", "dst_entity_id": "e1"}]
    touched = get_touched_entities(new_mentions, new_relationships, entity_id_to_index)
    assert set(touched) == {0, 1}


def test_get_touched_entities_unknown_id_ignored() -> None:
    entity_id_to_index = {"e1": 0}
    new_mentions = [{"entity_id": "e_unknown"}]
    touched = get_touched_entities(new_mentions, [], entity_id_to_index)
    assert touched == []
