#!/usr/bin/env python3
"""
End-to-end: ensure checkpoint exists, run LangGraph pipeline with demo events,
assert GNN features (embeddings, model_subgraph, embedding_centroid watchlist) when model runs.
Run from repo root: python scripts/run_gnn_e2e.py [--train] [--skip-train]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "apps" / "api"))

# Demo events in pipeline format (session_id, device_id, ts, seq, event_type, payload)
DEMO_EVENTS = [
    {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:00Z", "seq": 0, "event_type": "final_asr", "payload": {"text": "Someone from Medicare called", "confidence": 0.9}},
    {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:01Z", "seq": 1, "event_type": "intent", "payload": {"name": "share_ssn", "slots": {"number": "555-1234"}, "confidence": 0.85}},
    {"session_id": "s1", "device_id": "d1", "ts": "2024-01-15T10:00:02Z", "seq": 2, "event_type": "final_asr", "payload": {"text": "They said I need to verify immediately", "confidence": 0.88}},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GNN pipeline e2e: train (optional), run pipeline, assert features.")
    parser.add_argument("--train", action="store_true", help="Run ml.train to produce checkpoint if missing")
    parser.add_argument("--skip-train", action="store_true", help="Do not train; fail if checkpoint missing")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path (default: runs/hgt_baseline/best.pt)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or ROOT / "runs" / "hgt_baseline" / "best.pt"
    if not checkpoint.is_file():
        if args.skip_train:
            print(f"Checkpoint not found: {checkpoint}. Use --train or run: python -m ml.train --epochs 8")
            return 1
        if args.train:
            print("Training HGT baseline...")
            import subprocess
            r = subprocess.run(
                [sys.executable, "-m", "ml.train", "--epochs", "8", "--output", str(checkpoint.parent)],
                cwd=str(ROOT),
                timeout=120,
            )
            if r.returncode != 0:
                print("Training failed.")
                return 1
            print(f"Saved {checkpoint}")
        else:
            print(f"Checkpoint not found: {checkpoint}. Run with --train or: python -m ml.train --epochs 8")
            return 1

    # Run pipeline (use absolute path so it works from any cwd)
    import os
    os.environ["ANCHOR_ML_CHECKPOINT_PATH"] = str(checkpoint.resolve())

    from api.pipeline import run_pipeline

    print("Running pipeline with demo events...")
    state = run_pipeline(
        household_id="e2e-demo",
        ingested_events=DEMO_EVENTS,
        time_range_start="2024-01-15T00:00:00Z",
        time_range_end="2024-01-15T23:59:59Z",
    )

    # Assertions: when checkpoint exists and inference runs, we expect real GNN output
    model_available = state.get("_model_available", False)
    risk_scores = state.get("risk_scores", [])
    explanations = state.get("explanations", [])
    watchlists = state.get("watchlists", [])

    ok = True
    if not model_available:
        print("FAIL: _model_available is False (expected True when checkpoint exists)")
        ok = False
    else:
        print("OK: _model_available is True")

    embeddings_count = sum(1 for r in risk_scores if r.get("embedding") and len(r.get("embedding", [])) > 0)
    if model_available and embeddings_count == 0:
        print("FAIL: no embeddings on risk_scores (expected when model runs)")
        ok = False
    else:
        print(f"OK: {embeddings_count} risk_scores with embeddings")

    model_subgraph_count = sum(
        1 for e in explanations
        if (e.get("explanation_json") or {}).get("model_subgraph")
    )
    if model_available and len([r for r in risk_scores if r.get("score", 0) >= 0.4]) > 0 and model_subgraph_count == 0:
        print("FAIL: no model_subgraph in explanations (expected for nodes above threshold when model runs)")
        ok = False
    else:
        print(f"OK: {model_subgraph_count} explanations with model_subgraph")

    centroid_wls = [w for w in watchlists if w.get("watch_type") == "embedding_centroid"]
    # Centroid only when >= 3 high-risk with embeddings
    if model_available and embeddings_count >= 3 and len(centroid_wls) == 0:
        print("FAIL: no embedding_centroid watchlist (expected when >= 3 high-risk embeddings)")
        ok = False
    else:
        print(f"OK: {len(centroid_wls)} embedding_centroid watchlist(s) (need >=3 high-risk with emb for one)")

    if ok:
        print("\nAll GNN e2e checks passed.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
