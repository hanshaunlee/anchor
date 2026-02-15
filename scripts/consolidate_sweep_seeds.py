#!/usr/bin/env python3
"""
Consolidate embeddings entity meta, graph_stats, metrics, structural_report, and
all whitepaper CSVs for two sweep seeds into a single JSON file for the paper.

  python scripts/consolidate_sweep_seeds.py
  python scripts/consolidate_sweep_seeds.py --sweep-dir runs/structured_sweep_5seeds --seeds 0 1 --out runs/structured_sweep_5seeds/seed_0_seed_1_consolidated.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _load_csv_rows(p: Path, max_rows: int | None = None) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if max_rows is not None and i >= max_rows:
                break
            # normalize numeric strings to numbers where possible
            out = {}
            for k, v in row.items():
                try:
                    out[k] = float(v)
                except (ValueError, TypeError):
                    out[k] = v
            rows.append(out)
    return rows


def _load_csv_full(p: Path, max_rows: int | None = None) -> tuple[list[dict], int]:
    """Return (rows up to max_rows, total_row_count)."""
    if not p.exists():
        return [], 0
    rows = []
    total = 0
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        for i, row in enumerate(r):
            total = i + 1
            if max_rows is not None and i >= max_rows:
                continue
            out = {}
            for k, v in row.items():
                try:
                    out[k] = float(v)
                except (ValueError, TypeError):
                    out[k] = v
            rows.append(out)
    return rows, total


def consolidate_seed(run_dir: Path, *, cap_explanations: int = 5000) -> dict:
    """Gather graph_stats, metrics, structural_report, embeddings_entity_meta, whitepaper/* into one dict."""
    out = {
        "graph_stats": _load_json(run_dir / "graph_stats.json"),
        "metrics": _load_json(run_dir / "metrics.json"),
        "structural_report": _load_json(run_dir / "structural_report.json"),
        "embeddings_entity_meta": [],
        "embeddings_entity_meta_row_count": 0,
        "whitepaper": {},
    }
    # embeddings entity meta (can be hundreds of rows)
    emb_meta = run_dir / "embeddings_entity_meta.csv"
    if emb_meta.exists():
        with open(emb_meta, newline="") as f:
            r = csv.DictReader(f)
            out["embeddings_entity_meta"] = list(r)
            out["embeddings_entity_meta_row_count"] = len(out["embeddings_entity_meta"])
            for row in out["embeddings_entity_meta"]:
                for k in list(row):
                    try:
                        row[k] = float(row[k])
                    except (ValueError, TypeError):
                        pass

    wp = run_dir / "whitepaper"
    if wp.is_dir():
        # Small CSVs: include in full
        for name in ("loss.csv", "metrics_per_epoch.csv", "pr_curve_test.csv", "pr_curve_val.csv",
                     "roc_curve_test.csv", "roc_curve_val.csv", "y_true_y_score_test.csv", "y_true_y_score_val.csv",
                     "umap_points.csv", "motifs.csv"):
            path = wp / name
            if path.exists():
                out["whitepaper"][name.replace(".csv", "")] = _load_csv_rows(path)
        # explanations_edges can be 100k+ rows: include path, count, and sample
        exp_path = wp / "explanations_edges.csv"
        if exp_path.exists():
            sample, total = _load_csv_full(exp_path, max_rows=cap_explanations)
            out["whitepaper"]["explanations_edges_sample"] = sample
            out["whitepaper"]["explanations_edges_total_rows"] = total
            out["whitepaper"]["explanations_edges_path"] = str(exp_path)

    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate seed_0 and seed_1 sweep artifacts into one JSON")
    parser.add_argument("--sweep-dir", type=Path, default=Path("runs/structured_sweep_5seeds"), help="Sweep root")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="Seed indices")
    parser.add_argument("--out", type=Path, help="Output JSON path (default: <sweep_dir>/seed_0_seed_1_consolidated.json)")
    parser.add_argument("--cap-explanations", type=int, default=5000, help="Max explanation edges rows to embed (default 5000)")
    args = parser.parse_args()
    out_path = args.out or (args.sweep_dir / "seed_0_seed_1_consolidated.json")
    args.sweep_dir = args.sweep_dir.resolve()
    out_path = out_path.resolve()

    consolidated = {
        "sweep_dir": str(args.sweep_dir),
        "seeds": list(args.seeds),
        "seeds_data": {},
    }
    for seed in args.seeds:
        run_dir = args.sweep_dir / f"seed_{seed}"
        if not run_dir.is_dir():
            print(f"Warning: {run_dir} not found, skipping seed {seed}", file=sys.stderr)
            consolidated["seeds_data"][f"seed_{seed}"] = {"error": f"run_dir not found: {run_dir}"}
            continue
        consolidated["seeds_data"][f"seed_{seed}"] = consolidate_seed(run_dir, cap_explanations=args.cap_explanations)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
