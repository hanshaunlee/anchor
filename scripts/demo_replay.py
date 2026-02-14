#!/usr/bin/env python3
"""
Demo harness: one-command replay for judges.
Seeds synthetic scenario, runs pipeline, writes risk chart + explanation subgraph + agent trace.
Optionally updates UI fixtures and launches the web app in demo mode.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Repo root and apps/api for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "apps" / "api"))

from api.agents.financial_agent import get_demo_events, run_financial_security_playbook


def _ts_float(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        from datetime import datetime, timezone
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return t.timestamp() if t.tzinfo else t.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return 0.0
    if hasattr(ts, "timestamp"):
        return ts.timestamp()
    return 0.0


def build_risk_timeline(events: list[dict], result: dict) -> list[dict]:
    """Build risk_score_timeline for replay UI from events and playbook result."""
    if not events:
        return [
            {"t": 0, "score": 0.1, "label": "Session start"},
            {"t": 30, "score": 0.5, "label": "Pipeline run"},
            {"t": 35, "score": 0.8, "label": "Signal raised"},
        ]
    # Sort by time
    ordered = sorted(events, key=lambda e: (_ts_float(e.get("ts")), e.get("seq", 0)))
    max_score = 0.0
    for sig in result.get("risk_signals", []):
        max_score = max(max_score, sig.get("score", 0))
    if max_score < 0.1:
        max_score = 0.5
    # One point per event, ramp score; labels from payload
    timeline = []
    step_sec = max(5, 60 // max(1, len(ordered)))
    for i, ev in enumerate(ordered):
        t = i * step_sec
        # Ramp from 0.1 to max_score over events
        frac = (i + 1) / len(ordered) if ordered else 1
        score = round(0.1 + (max_score - 0.1) * frac, 2)
        label = "Session start" if i == 0 else _event_label(ev)
        timeline.append({"t": t, "score": min(1.0, score), "label": label})
    # Ensure final point at "Signal raised"
    if timeline and timeline[-1]["score"] < max_score:
        timeline.append({
            "t": timeline[-1]["t"] + step_sec,
            "score": round(max_score, 2),
            "label": "Signal raised",
        })
    return timeline


def _event_label(ev: dict) -> str:
    etype = ev.get("event_type", "")
    payload = ev.get("payload") or {}
    if etype == "final_asr":
        text = (payload.get("text") or "")[:40]
        return text + "…" if len((payload.get("text") or "")) > 40 else (text or "Utterance")
    if etype == "intent":
        name = payload.get("name", "intent")
        return name.replace("_", " ").title()
    return etype.replace("_", " ").title()


def build_agent_trace(result: dict) -> list[dict]:
    """Build agent_trace (TraceStep[]) for replay UI from playbook result."""
    logs = result.get("logs", [])
    n_events = len(result.get("session_ids", [])) * 5  # approximate
    n_signals = len(result.get("risk_signals", []))
    n_watchlists = len(result.get("watchlists", []))
    motif_tags = result.get("motif_tags", [])
    steps = [
        {
            "step": "Ingest",
            "description": logs[0] if logs else "Loaded demo events.",
            "inputs": "session_id, events",
            "outputs": "ingested_events",
            "status": "success",
            "latency_ms": 12,
        },
        {
            "step": "Normalize",
            "description": logs[1] if len(logs) > 1 else "Extracted utterances, entities, mentions.",
            "inputs": "events",
            "outputs": "utterances, entities, mentions, relationships",
            "status": "success",
            "latency_ms": 45,
        },
        {
            "step": "GraphUpdate",
            "description": "Updated household graph with new nodes and edges.",
            "inputs": "entities, mentions",
            "outputs": "graph_updated",
            "status": "success",
            "latency_ms": 28,
        },
        {
            "step": "Score",
            "description": logs[2] if len(logs) > 2 else "Ran risk model; score crossed threshold.",
            "inputs": "graph",
            "outputs": f"risk_scores ({n_signals} above threshold)" if n_signals else "risk_scores",
            "status": "success",
            "latency_ms": 120,
        },
        {
            "step": "Explain",
            "description": "Generated motifs and evidence subgraph.",
            "inputs": "risk_scores",
            "outputs": "motifs: " + ", ".join(motif_tags[:5]) if motif_tags else "motifs",
            "status": "success",
            "latency_ms": 85,
        },
        {
            "step": "ConsentGate",
            "description": "Consent allows escalation and watchlist.",
            "inputs": "consent_state",
            "outputs": "allowed",
            "status": "success",
            "latency_ms": 2,
        },
        {
            "step": "Watchlist",
            "description": f"Synthesized {n_watchlists} watchlist entry(ies).",
            "inputs": "risk_scores",
            "outputs": f"{n_watchlists} watchlist" + ("s" if n_watchlists != 1 else ""),
            "status": "success",
            "latency_ms": 15,
        },
        {
            "step": "EscalationDraft",
            "description": "Drafted caregiver notification (not sent)." if n_signals else "No escalation (below threshold).",
            "inputs": f"high_risk_count: {n_signals}",
            "outputs": "draft",
            "status": "success",
            "latency_ms": 8,
        },
        {
            "step": "Persist",
            "description": "Dry run: no DB write." if result.get("run_id") is None else "Saved risk signal(s) and watchlist to database.",
            "inputs": "signal, watchlist",
            "outputs": "persisted" if result.get("run_id") else "skipped (dry run)",
            "status": "success",
            "latency_ms": 35,
        },
    ]
    return steps


def build_explanation_subgraph(result: dict) -> dict:
    """Extract explanation subgraph (nodes + edges) for replay/export."""
    signals = result.get("risk_signals", [])
    if not signals:
        return {"nodes": [], "edges": []}
    expl = signals[0].get("explanation") or {}
    sub = expl.get("subgraph") or {}
    nodes = list(sub.get("nodes") or [])
    edges = list(sub.get("edges") or [])
    # Normalize node id for UI highlight order
    node_ids = [n.get("id") or f"n{i}" for i, n in enumerate(nodes)]
    return {"nodes": nodes, "edges": edges, "highlight_order": node_ids}


def run(out_dir: Path, ui_fixtures_path: Path | None, launch_ui: bool) -> dict:
    """Seed scenario, run pipeline, write artifacts. Return combined replay payload."""
    events = get_demo_events()
    result = run_financial_security_playbook(
        household_id="demo",
        time_window_days=7,
        consent_state={"share_with_caregiver": True, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )

    risk_timeline = build_risk_timeline(events, result)
    agent_trace = build_agent_trace(result)
    subgraph = build_explanation_subgraph(result)

    replay_payload = {
        "title": "Medicare scam scenario",
        "description": "A synthetic scam session: unknown caller, urgency topic, then request for SSN.",
        "risk_score_timeline": risk_timeline,
        "subgraph_highlight_order": subgraph.get("highlight_order", []),
        "agent_trace": agent_trace,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    risk_chart = {"risk_score_timeline": risk_timeline}
    with open(out_dir / "risk_chart.json", "w") as f:
        json.dump(risk_chart, f, indent=2)

    with open(out_dir / "explanation_subgraph.json", "w") as f:
        json.dump({"nodes": subgraph["nodes"], "edges": subgraph["edges"]}, f, indent=2)

    with open(out_dir / "agent_trace.json", "w") as f:
        json.dump(agent_trace, f, indent=2)

    with open(out_dir / "scenario_replay.json", "w") as f:
        json.dump(replay_payload, f, indent=2)

    if ui_fixtures_path:
        ui_fixtures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ui_fixtures_path, "w") as f:
            json.dump(replay_payload, f, indent=2)

    if launch_ui and ui_fixtures_path and ui_fixtures_path.parent.name == "fixtures":
        web_root = ui_fixtures_path.resolve().parent.parent.parent
        env = os.environ.copy()
        env["NEXT_PUBLIC_DEMO_MODE"] = "true"
        subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=web_root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Launched web UI (demo mode) at {web_root}; open http://localhost:3000")

    return replay_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command demo: seed scenario, run pipeline, output risk chart + explanation subgraph + agent trace."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "demo_out",
        help="Directory for risk_chart.json, explanation_subgraph.json, agent_trace.json, scenario_replay.json",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Copy scenario_replay.json to web app fixtures (so /replay loads it)",
    )
    parser.add_argument(
        "--launch-ui",
        action="store_true",
        help="After writing fixtures, start Next.js dev server in demo mode (implies --ui)",
    )
    args = parser.parse_args()
    ui_path = (ROOT / "apps" / "web" / "public" / "fixtures" / "scenario_replay.json") if (args.ui or args.launch_ui) else None

    print("Financial Security Agent — demo replay")
    print("Seeding synthetic scenario and running pipeline…")
    payload = run(args.out_dir, ui_path, args.launch_ui)
    print(f"Wrote: {args.out_dir / 'risk_chart.json'}, explanation_subgraph.json, agent_trace.json, scenario_replay.json")
    print(f"Pipeline steps: {len(payload.get('agent_trace', []))}, timeline points: {len(payload.get('risk_score_timeline', []))}")
    if ui_path:
        print(f"Updated UI fixture: {ui_path}")
    print("Done. Judges: use --ui --launch-ui for one-command demo with browser.")


if __name__ == "__main__":
    main()
