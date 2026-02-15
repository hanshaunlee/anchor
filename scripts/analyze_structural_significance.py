#!/usr/bin/env python3
"""
Report structural significance for a structured-synthetic run: Cohen's d, t-tests, degree distribution.

Usage:
  python scripts/analyze_structural_significance.py runs/hgt_structured_02
  python scripts/analyze_structural_significance.py runs/hgt_structured_02 -o runs/hgt_structured_02/structural_report.json

Reads data_structured_synthetic.json and graph_stats.json (and optionally metrics.json) from the run dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_run(run_dir: Path) -> tuple[dict, dict, dict | None]:
    run_dir = Path(run_dir)
    data_path = run_dir / "data_structured_synthetic.json"
    stats_path = run_dir / "graph_stats.json"
    metrics_path = run_dir / "metrics.json"
    if not data_path.exists():
        # Fallback: graph_stats may have structured_stats inside
        if not stats_path.exists():
            raise SystemExit(f"Run dir must contain data_structured_synthetic.json or graph_stats.json: {run_dir}")
        with open(stats_path) as f:
            graph_stats = json.load(f)
        structured = graph_stats.get("structured_stats") or {}
    else:
        with open(data_path) as f:
            structured = json.load(f)
        graph_stats = json.load(stats_path.open()) if stats_path.exists() else {}
    metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    return structured, graph_stats, metrics


def build_report(structured: dict, graph_stats: dict, metrics: dict | None) -> dict:
    report: dict = {
        "structural_significance": {},
        "degree_distribution": {},
        "motif_and_structure": {},
        "model_metrics": {},
    }

    # Cohen's d for all structural metrics (incl. motif-like: triangle_count, k_core, etc.)
    cohen_d = structured.get("cohen_d") or {}
    t_test_pvalue = structured.get("t_test_pvalue") or {}
    for name, d in cohen_d.items():
        report["structural_significance"][name] = {
            "cohen_d": round(d, 4),
            "t_test_pvalue": t_test_pvalue.get(name),
            "interpretation": "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small",
        }

    # Degree distribution: t-test, Cohen's d, KS
    report["degree_distribution"] = {
        "mean_degree_fraud": structured.get("mean_degree_fraud"),
        "mean_degree_normal": structured.get("mean_degree_normal"),
        "degree_cohen_d": structured.get("degree_cohen_d"),
        "degree_ttest_pvalue": structured.get("degree_ttest_pvalue"),
        "degree_ks_statistic": structured.get("degree_ks_statistic"),
        "degree_ks_pvalue": structured.get("degree_ks_pvalue"),
    }

    # Motif / aggregate (from graph_stats or motifs.json)
    report["motif_and_structure"]["label_prevalence"] = structured.get("label_prevalence")
    report["motif_and_structure"]["mean_triangle_fraud"] = structured.get("mean_triangle_fraud")
    report["motif_and_structure"]["mean_triangle_normal"] = structured.get("mean_triangle_normal")
    report["motif_and_structure"]["mean_cross_session_ratio_fraud"] = structured.get("mean_cross_session_ratio_fraud")
    report["motif_and_structure"]["mean_cross_session_ratio_normal"] = structured.get("mean_cross_session_ratio_normal")
    report["motif_and_structure"]["modularity"] = structured.get("modularity")
    report["motif_and_structure"]["k_core_dist_fraud"] = structured.get("k_core_dist_fraud")
    report["motif_and_structure"]["k_core_dist_normal"] = structured.get("k_core_dist_normal")

    if metrics:
        report["model_metrics"] = {
            "test_roc_auc": metrics.get("test_roc_auc"),
            "test_pr_auc": metrics.get("test_pr_auc"),
            "val_roc_auc": metrics.get("val_roc_auc"),
            "val_pr_auc": metrics.get("val_pr_auc"),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze structural significance of a structured-synthetic run")
    parser.add_argument("run_dir", type=Path, help="Run directory (e.g. runs/hgt_structured_02)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write report JSON here")
    parser.add_argument("--no-print", action="store_true", help="Do not print report to stdout")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else _REPO_ROOT / args.run_dir

    structured, graph_stats, metrics = load_run(run_dir)
    report = build_report(structured, graph_stats, metrics)

    if args.output:
        out_path = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {out_path}", file=sys.stderr)

    if not args.no_print:
        print("=== Structural significance (Cohen's d, t-tests) ===")
        for name, v in report["structural_significance"].items():
            print(f"  {name}: Cohen's d = {v['cohen_d']} ({v['interpretation']}), t-test p = {v['t_test_pvalue']}")
        print("\n=== Degree distribution ===")
        for k, v in report["degree_distribution"].items():
            print(f"  {k}: {v}")
        print("\n=== Motif / structure (means) ===")
        for k, v in report["motif_and_structure"].items():
            print(f"  {k}: {v}")
        if report["model_metrics"]:
            print("\n=== Model metrics ===")
            for k, v in report["model_metrics"].items():
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
