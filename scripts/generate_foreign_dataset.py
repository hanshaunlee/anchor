#!/usr/bin/env python3
"""
Generate an extremely extensive synthetic foreign dataset for GNN training.
Same output format as scripts/export_supabase_for_gnn.py (graph.pt, meta.json, graph_stats.json)
so you can train with: --dataset supabase_export --data-dir <output>.

Use this as "foreign" data: train here, then run inference on real user graphs with the checkpoint.

Usage:
  python scripts/generate_foreign_dataset.py --output data/foreign_large --seed 42
  python -m ml.train --config ml/configs/hgt_baseline.yaml --dataset supabase_export \\
    --data-dir data/foreign_large --run-dir runs/hgt_foreign_01 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "apps" / "api"))

from ml.graph.builder import build_hetero_from_tables
from ml.train import _add_synthetic_masks_and_labels, _graph_stats

INTENTS = ("call", "reminder", "transfer", "balance", "pay_bill", "alerts", "watchlist", "other")


def generate_foreign_dataset(
    num_entities: int,
    num_sessions: int,
    utterances_per_session: int,
    entities_per_utterance_min: int,
    entities_per_utterance_max: int,
    seed: int,
    max_co_occurs: int | None = 120_000,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Generate sessions, utterances, entities, mentions, relationships (CO_OCCURS)."""
    rng = np.random.default_rng(seed)
    base_ts = datetime.now(timezone.utc) - timedelta(days=365)
    entity_ids = [f"e_{i}" for i in range(num_entities)]
    entities = [{"id": eid} for eid in entity_ids]
    entity_set = set(entity_ids)

    sessions = []
    utterances = []
    mentions = []
    # Session -> set of entity_ids mentioned in that session (for CO_OCCURS)
    session_entities: list[set[str]] = []

    for s in range(num_sessions):
        sid = f"s_{s}"
        started_at = (base_ts + timedelta(hours=s * 2)).isoformat()
        sessions.append({"id": sid, "started_at": started_at})
        session_ent_set: set[str] = set()
        n_utt = int(rng.integers(
            max(1, utterances_per_session - 20),
            utterances_per_session + 20,
            endpoint=True,
        ))
        for u in range(n_utt):
            uid = f"u_{s}_{u}"
            utterances.append({
                "id": uid,
                "session_id": sid,
                "intent": str(rng.choice(INTENTS)),
                "ts": (base_ts + timedelta(hours=s * 2, minutes=u)).isoformat(),
            })
            n_ent = rng.integers(entities_per_utterance_min, entities_per_utterance_max + 1, endpoint=True)
            chosen = rng.choice(entity_ids, size=min(n_ent, len(entity_ids)), replace=False)
            for eid in chosen:
                mentions.append({"utterance_id": uid, "entity_id": eid})
                session_ent_set.add(eid)
        session_entities.append(session_ent_set)

    # Build CO_OCCURS from session co-occurrence (all pairs within each session)
    # Cap total edges to avoid NaN/instability on huge graphs while still being very large
    seen_pairs: set[tuple[str, str]] = set()
    relationships = []
    t0 = base_ts.timestamp()
    for sess_ents in session_entities:
        if max_co_occurs is not None and len(relationships) >= max_co_occurs:
            break
        lst = sorted(sess_ents)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if max_co_occurs is not None and len(relationships) >= max_co_occurs:
                    break
                a, b = lst[i], lst[j]
                if a > b:
                    a, b = b, a
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                ts_mid = t0 + rng.uniform(0, 86400 * 365)
                relationships.append({
                    "src_entity_id": a,
                    "dst_entity_id": b,
                    "rel_type": "CO_OCCURS",
                    "first_seen_at": ts_mid - 3600,
                    "last_seen_at": ts_mid,
                    "count": int(rng.integers(1, 10)),
                    "weight": float(rng.uniform(0.5, 1.5)),
                })

    return sessions, utterances, entities, mentions, relationships


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate extensive foreign dataset for GNN training")
    parser.add_argument("--output", type=Path, default=Path("data/foreign_large"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-entities", type=int, default=2500, help="Number of entities")
    parser.add_argument("--num-sessions", type=int, default=1000, help="Number of sessions")
    parser.add_argument("--utterances-per-session", type=int, default=60, help="Approx utterances per session")
    parser.add_argument("--entities-per-utterance-min", type=int, default=1)
    parser.add_argument("--entities-per-utterance-max", type=int, default=4)
    parser.add_argument("--max-co-occurs", type=int, default=120_000,
                        help="Cap CO_OCCURS edges for training stability (default 120k)")
    args = parser.parse_args()

    sessions, utterances, entities, mentions, relationships = generate_foreign_dataset(
        num_entities=args.num_entities,
        num_sessions=args.num_sessions,
        utterances_per_session=args.utterances_per_session,
        entities_per_utterance_min=args.entities_per_utterance_min,
        entities_per_utterance_max=args.entities_per_utterance_max,
        seed=args.seed,
        max_co_occurs=args.max_co_occurs,
    )
    entity_ids = [e["id"] for e in entities]
    n_edges = sum(1 for r in relationships if r.get("rel_type") == "CO_OCCURS")
    print(f"Generated: {len(entities)} entities, {len(sessions)} sessions, {len(utterances)} utterances, "
          f"{len(mentions)} mentions, {len(relationships)} relationships ({n_edges} CO_OCCURS)")

    household_id = "foreign"
    devices: list[dict] = []
    data = build_hetero_from_tables(
        household_id,
        sessions,
        utterances,
        entities,
        mentions,
        relationships,
        devices,
        events=None,
    )
    try:
        node_types, edge_types = data.metadata()
    except Exception:
        node_types = [getattr(s, "key", i) for i, s in enumerate(data.node_stores)]
        edge_types = [getattr(s, "key", None) for s in data.edge_stores]
        edge_types = [e for e in edge_types if e is not None]
    in_channels = {nt: data[nt].x.size(1) for nt in node_types}
    _add_synthetic_masks_and_labels(data, "entity", seed=args.seed)
    graph_stats = _graph_stats(data, node_types, edge_types)
    graph_stats["num_entities"] = len(entities)
    graph_stats["num_entity_edges"] = n_edges

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output / "graph.pt")
    meta = {
        "in_channels": in_channels,
        "entity_ids": entity_ids,
        "node_types": node_types,
        "edge_types": [list(e) for e in edge_types],
        "household_id": household_id,
        "source": "generate_foreign_dataset",
        "num_entities": len(entities),
        "num_sessions": len(sessions),
        "num_utterances": len(utterances),
        "num_mentions": len(mentions),
        "num_relationships": len(relationships),
        "seed": args.seed,
    }
    with open(args.output / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.output / "graph_stats.json", "w") as f:
        json.dump(graph_stats, f, indent=2)
    print(f"Saved graph.pt, meta.json, graph_stats.json to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
