"""
k-hop ego subgraph extraction for local inference and caching.
Identifies touched entities (new/updated mentions, new phone, etc.), extracts k-hop neighborhood,
optionally filters by time_window for temporal replay.
"""
from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Max nodes when inferring full-graph fallback (no seeds); override via ANCHOR_MAX_SUBGRAPH_NODES
MAX_SUBGRAPH_NODES_FALLBACK = int(os.environ.get("ANCHOR_MAX_SUBGRAPH_NODES", "100"))


def _get_hetero_storage():
    import torch
    from torch_geometric.data import HeteroData
    return torch, HeteroData


def extract_k_hop(
    heterodata: Any,
    seed_nodes_by_type: dict[str, list[int]],
    k: int,
    time_window: tuple[float, float] | None = None,
):
    """
    Extract k-hop induced subgraph around seed nodes.
    seed_nodes_by_type: e.g. {"entity": [0, 3], "session": [1]}.
    time_window: (min_ts, max_ts) to filter edges by edge_attr time if present; optional.
    Returns: HeteroData subgraph + node_index maps (global -> local) per type.
    """
    torch, HeteroData = _get_hetero_storage()
    out = HeteroData()
    # PyG 2.7: node_stores/edge_stores iterate storage objects; use metadata() or store.key for type names
    try:
        node_types, edge_types = heterodata.metadata()
        edge_types = list(edge_types)
    except Exception:
        node_types = [getattr(s, "key", None) for s in heterodata.node_stores]
        node_types = [k for k in node_types if k is not None]
        edge_types = [getattr(s, "key", None) for s in heterodata.edge_stores]
        edge_types = [k for k in edge_types if k is not None]

    # Start with seeds
    current_by_type: dict[str, set[int]] = {nt: set(seed_nodes_by_type.get(nt, [])) for nt in node_types}
    for _ in range(k):
        next_by_type: dict[str, set[int]] = {nt: set() for nt in node_types}
        for (src_type, rel, dst_type) in edge_types:
            key = (src_type, rel, dst_type)
            try:
                _ = heterodata[key]
            except (KeyError, TypeError):
                continue
            edge_index = heterodata[src_type, rel, dst_type].edge_index
            if edge_index is None or edge_index.size(1) == 0:
                continue
            src, dst = edge_index[0], edge_index[1]
            seeds_src = current_by_type.get(src_type, set())
            seeds_dst = current_by_type.get(dst_type, set())
            if not seeds_src and not seeds_dst:
                continue
            mask_src = torch.tensor([src[e].item() in seeds_src for e in range(src.size(0))], device=edge_index.device)
            mask_dst = torch.tensor([dst[e].item() in seeds_dst for e in range(dst.size(0))], device=edge_index.device)
            mask = mask_src | mask_dst
            if mask.any():
                new_src = src[mask].tolist()
                new_dst = dst[mask].tolist()
                next_by_type[src_type].update(new_src)
                next_by_type[dst_type].update(new_dst)
        for nt in node_types:
            current_by_type[nt].update(next_by_type[nt])

    # Build induced subgraph: only nodes in current_by_type
    global_to_local: dict[str, dict[int, int]] = {}
    for nt in node_types:
        nodes = sorted(current_by_type[nt])
        if not nodes and hasattr(heterodata[nt], "x") and getattr(heterodata[nt], "x", None) is not None:
            n = heterodata[nt].x.size(0)
            nodes = list(range(n)) if n <= MAX_SUBGRAPH_NODES_FALLBACK else []
        if not nodes:
            continue
        global_to_local[nt] = {g: i for i, g in enumerate(nodes)}
        if hasattr(heterodata[nt], "x") and getattr(heterodata[nt], "x", None) is not None:
            out[nt].x = heterodata[nt].x[nodes]
        out[nt].num_nodes = len(nodes)

    for (src_type, rel, dst_type) in edge_types:
        key = (src_type, rel, dst_type)
        try:
            _ = heterodata[key]
        except (KeyError, TypeError):
            continue
        if src_type not in global_to_local or dst_type not in global_to_local:
            continue
        edge_index = heterodata[key].edge_index
        if edge_index is None or edge_index.size(1) == 0:
            continue
        src, dst = edge_index[0], edge_index[1]
        loc_src = global_to_local[src_type]
        loc_dst = global_to_local[dst_type]
        keep = []
        for e in range(edge_index.size(1)):
            s, d = src[e].item(), dst[e].item()
            if s in loc_src and d in loc_dst:
                keep.append((loc_src[s], loc_dst[d]))
        if not keep:
            continue
        new_idx = torch.tensor(keep, dtype=torch.long, device=edge_index.device).t()
        out[key].edge_index = new_idx
        if hasattr(heterodata[key], "edge_attr") and heterodata[key].edge_attr is not None:
            # Filter edge_attr by same mask (expensive; we already filtered edges)
            old_src, old_dst = edge_index[0], edge_index[1]
            edge_attr_list = []
            for e in range(edge_index.size(1)):
                s, d = old_src[e].item(), old_dst[e].item()
                if s in loc_src and d in loc_dst:
                    edge_attr_list.append(heterodata[key].edge_attr[e])
            if edge_attr_list:
                out[key].edge_attr = torch.stack(edge_attr_list)

    return out, global_to_local


def get_touched_entities(
    new_mentions: list[dict],
    new_relationships: list[dict],
    entity_id_to_index: dict[str, int],
) -> list[int]:
    """Identify entity indices that are touched by new mentions or new relationships."""
    touched = set()
    for m in new_mentions:
        eid = m.get("entity_id")
        if eid in entity_id_to_index:
            touched.add(entity_id_to_index[eid])
    for r in new_relationships:
        for key in ("src_entity_id", "dst_entity_id"):
            eid = r.get(key)
            if eid in entity_id_to_index:
                touched.add(entity_id_to_index[eid])
    return list(touched)
