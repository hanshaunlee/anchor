#!/usr/bin/env python3
"""
Run the structured-synthetic 5-seed sweep locally (no Modal).
Produces runs/structured_sweep_5seeds/seed_{0..4}/ and summary.json with mean ± std ROC-AUC, PR-AUC.

  python scripts/run_structured_sweep_local.py --n-entities 2500 --epochs 100
  python scripts/run_structured_sweep_local.py --n-entities 500 --epochs 30 --n-seeds 2   # quick check
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "runs" / "structured_sweep_5seeds"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-entities", type=int, default=2500)
    parser.add_argument("--n-sessions", type=int, default=None, help="Default: min(400, max(180, n_entities//3))")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    n_sessions = args.n_sessions
    if n_sessions is None:
        n_sessions = min(400, max(180, args.n_entities // 3))

    os.environ["ANCHOR_STRUCTURED_N_ENTITIES"] = str(args.n_entities)
    os.environ["ANCHOR_STRUCTURED_N_SESSIONS"] = str(n_sessions)

    results = []
    for seed in range(args.n_seeds):
        run_dir = SWEEP_DIR / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable, "-m", "ml.train",
            "--config", "ml/configs/hgt_baseline.yaml",
            "--dataset", "structured_synthetic",
            "--run-dir", str(run_dir),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--device", args.device,
        ]
        print(f"Seed {seed}: running {' '.join(argv)}", flush=True)
        ret = subprocess.run(argv, cwd=REPO_ROOT)
        if ret.returncode != 0:
            results.append({"seed": seed, "run_dir": str(run_dir), "error": f"exit {ret.returncode}"})
            continue
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            results.append({
                "seed": seed,
                "run_dir": str(run_dir),
                "test_roc_auc": m.get("test_roc_auc"),
                "test_pr_auc": m.get("test_pr_auc"),
                "val_roc_auc": m.get("val_roc_auc"),
                "val_pr_auc": m.get("val_pr_auc"),
            })
        else:
            results.append({"seed": seed, "run_dir": str(run_dir), "error": "metrics.json not found"})

    import numpy as np
    roc = [r["test_roc_auc"] for r in results if r.get("test_roc_auc") is not None]
    pr = [r["test_pr_auc"] for r in results if r.get("test_pr_auc") is not None]
    summary = {
        "n_entities": args.n_entities,
        "n_sessions": n_sessions,
        "epochs": args.epochs,
        "n_seeds": args.n_seeds,
        "per_seed": results,
        "test_roc_auc_mean": float(np.mean(roc)) if roc else None,
        "test_roc_auc_std": float(np.std(roc)) if len(roc) > 1 else None,
        "test_pr_auc_mean": float(np.mean(pr)) if pr else None,
        "test_pr_auc_std": float(np.std(pr)) if len(pr) > 1 else None,
        "val_roc_auc_mean": float(np.mean([r["val_roc_auc"] for r in results if r.get("val_roc_auc") is not None])) if results else None,
        "val_pr_auc_mean": float(np.mean([r["val_pr_auc"] for r in results if r.get("val_pr_auc") is not None])) if results else None,
    }
    summary_path = SWEEP_DIR / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print("Summary:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
