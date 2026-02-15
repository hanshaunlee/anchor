#!/usr/bin/env python3
"""
Export Supabase events → normalized graph → PyG HeteroData for GNN training.

Usage:
  Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (e.g. in .env or apps/api/.env), then:
    python scripts/export_supabase_for_gnn.py --household-id <uuid> --output data/supabase_export
    python -m ml.train --config ml/configs/hgt_baseline.yaml --dataset supabase_export --data-dir data/supabase_export --run-dir runs/hgt_supabase_01

Options:
  --household-id   Household UUID (required).
  --time-window-days  Days of events to fetch (default: 90).
  --max-events     Cap events (default: 50000).
  --max-sessions   Cap sessions (default: 2000).
  --output         Directory to write graph.pt and meta.json (default: data/supabase_export).
  --seed           Random seed for train/val/test split (default: 42).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Run from repo root; domain is under apps/api
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "apps" / "api"))

import torch


def _load_dotenv() -> None:
    """Load .env from apps/api or repo root so SUPABASE_* are set when running from CLI."""
    for env_path in (_REPO_ROOT / "apps" / "api" / ".env", _REPO_ROOT / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if "#" in v and not v.startswith("'"):
                v = v.split("#")[0].strip()
            os.environ.setdefault(k.strip(), v)


def fetch_events(supabase, household_id: str, time_window_days: int, max_events: int, max_sessions: int) -> tuple[list[dict], list[dict]]:
    """Fetch events and sessions from Supabase. Returns (events, sessions)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=time_window_days)
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    sess_r = (
        supabase.table("sessions")
        .select("id, started_at, device_id")
        .eq("household_id", household_id)
        .gte("started_at", start_iso)
        .lte("started_at", end_iso)
        .order("started_at", desc=False)
        .limit(max_sessions)
        .execute()
    )
    sessions = list(sess_r.data or [])
    session_ids = [s["id"] for s in sessions]
    if not session_ids:
        return [], []

    events = []
    for sid in session_ids:
        ev_r = (
            supabase.table("events")
            .select("id, session_id, device_id, ts, seq, event_type, payload")
            .eq("session_id", sid)
            .order("ts")
            .limit(2000)
            .execute()
        )
        events.extend(ev_r.data or [])
        if len(events) >= max_events:
            break
    events = events[:max_events]
    # Normalize to pipeline shape
    out_ev = []
    for ev in events:
        out_ev.append({
            "session_id": ev.get("session_id"),
            "device_id": ev.get("device_id"),
            "ts": ev.get("ts"),
            "seq": ev.get("seq", 0),
            "event_type": ev.get("event_type", ""),
            "payload": ev.get("payload") or {},
        })
    return out_ev, sessions


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Supabase graph for GNN training")
    parser.add_argument("--household-id", type=str, required=True, help="Household UUID")
    parser.add_argument("--time-window-days", type=int, default=90)
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--max-sessions", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("data/supabase_export"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _load_dotenv()
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (e.g. in .env or apps/api/.env) to export from Supabase.", file=sys.stderr)
        return 1

    from supabase import create_client
    supabase = create_client(url, key)

    from domain.graph_service import build_graph_from_events
    from ml.graph.builder import build_hetero_from_tables

    events, sessions = fetch_events(
        supabase, args.household_id, args.time_window_days, args.max_events, args.max_sessions
    )
    if not events:
        print("No events found for household in time window.", file=sys.stderr)
        return 1

    utterances, entities, mentions, relationships = build_graph_from_events(
        args.household_id, events, supabase=None
    )
    entity_ids = [e.get("id") for e in entities if e.get("id")]
    n_entities = len(entity_ids)
    n_edges = sum(1 for r in relationships if r.get("rel_type") == "CO_OCCURS")
    print(f"Normalized: {len(utterances)} utterances, {n_entities} entities, {len(relationships)} relationships ({n_edges} co_occurs)")

    # Sessions as list of dicts with id (and started_at for builder)
    session_list = [{"id": str(s["id"]), "started_at": s.get("started_at")} for s in sessions]
    devices: list[dict] = []
    data = build_hetero_from_tables(
        args.household_id,
        session_list,
        utterances,
        entities,
        mentions,
        relationships,
        devices,
        events=None,
    )

    # Add train/val/test masks and labels (seed-based) so training can use the graph
    from ml.train import _add_synthetic_masks_and_labels, _graph_stats
    try:
        node_types, edge_types = data.metadata()
    except Exception:
        node_types = [getattr(s, "key", i) for i, s in enumerate(data.node_stores)]
        edge_types = [getattr(s, "key", None) for s in data.edge_stores]
        edge_types = [e for e in edge_types if e is not None]
    if "entity" in node_types and data["entity"].x.size(0) > 0:
        _add_synthetic_masks_and_labels(data, "entity", seed=args.seed)
    graph_stats = _graph_stats(data, node_types, edge_types)
    graph_stats["num_entities"] = n_entities
    graph_stats["num_entity_edges"] = n_edges

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output / "graph.pt")
    in_channels = {nt: data[nt].x.size(1) for nt in node_types}
    meta = {
        "in_channels": in_channels,
        "entity_ids": entity_ids,
        "node_types": node_types,
        "edge_types": [list(e) for e in edge_types],
        "household_id": args.household_id,
        "time_window_days": args.time_window_days,
        "num_events": len(events),
        "num_sessions": len(sessions),
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
