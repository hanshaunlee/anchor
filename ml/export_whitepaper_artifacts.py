#!/usr/bin/env python3
"""
Turn run logs into plot-ready CSVs for LaTeX (pgfplots).
Reads history.jsonl, preds_*.npz, embeddings*.npz, motifs.json, explanations;
writes run_dir/whitepaper/ with:
  - loss.csv, metrics_per_epoch.csv (per-epoch train loss + val/test PR-AUC, ROC-AUC)
  - y_true_y_score_test.csv, y_true_y_score_val.csv (raw labels + scores for threshold tables, calibration, histograms)
  - pr_curve_*.csv, roc_curve_*.csv, umap_points.csv, motifs.csv, explanations_edges.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export whitepaper plot artifacts from a run directory")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory (e.g. runs/hgt_synth_01)")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    out_dir = run_dir / "whitepaper"
    out_dir.mkdir(parents=True, exist_ok=True)

    # history.jsonl -> loss.csv + metrics_per_epoch.csv (per-epoch val/test PR-AUC, ROC-AUC)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        by_epoch: dict[int, dict] = {}
        with open(history_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                e = r.get("epoch")
                if e is not None:
                    by_epoch[int(e)] = r
        if by_epoch:
            with open(out_dir / "loss.csv", "w") as f:
                f.write("epoch,train_loss\n")
                for e in sorted(by_epoch.keys()):
                    r = by_epoch[e]
                    loss = r.get("train_loss")
                    if loss is not None:
                        f.write(f"{e},{loss}\n")
            logger.info("Wrote %s (%d epochs)", out_dir / "loss.csv", len(by_epoch))
            # Per-epoch metrics for curves (val/test PR-AUC, ROC-AUC)
            metric_cols = ["epoch", "train_loss", "val_pr_auc", "val_roc_auc", "test_pr_auc", "test_roc_auc"]
            with open(out_dir / "metrics_per_epoch.csv", "w") as f:
                f.write(",".join(metric_cols) + "\n")
                for e in sorted(by_epoch.keys()):
                    r = by_epoch[e]
                    row = [str(r.get(c, "")) for c in metric_cols]
                    f.write(",".join(row) + "\n")
            logger.info("Wrote metrics_per_epoch.csv (%d epochs)", len(by_epoch))

    # preds_*.npz -> pr_curve.csv, roc_curve.csv; raw y_true,y_score for threshold tables, calibration, histograms
    for name in ("test", "val"):
        preds_path = run_dir / f"preds_{name}.npz"
        if not preds_path.exists():
            continue
        data = np.load(preds_path, allow_pickle=True)
        y_true = np.asarray(data["y_true"]).ravel()
        probs = data["probs"] if "probs" in data else (data["logits"][:, 1] if data["logits"].ndim == 2 else data["logits"])
        probs = np.asarray(probs).ravel()
        valid = y_true >= 0
        if valid.any():
            y_true = y_true[valid]
            probs = probs[valid]
        # Raw y_true + y_score (probability of positive class) for recall@k, precision@k, calibration, score histograms
        has_entity_ids = "entity_ids" in data and len(data["entity_ids"]) == len(y_true)
        raw_path = out_dir / f"y_true_y_score_{name}.csv"
        with open(raw_path, "w") as f:
            if has_entity_ids:
                f.write("y_true,y_score,entity_id\n")
                for i in range(len(y_true)):
                    eid = getattr(data["entity_ids"][i], "item", lambda: data["entity_ids"][i])()
                    f.write(f"{int(y_true[i])},{probs[i]:.6f},{eid}\n")
            else:
                f.write("y_true,y_score\n")
                for i in range(len(y_true)):
                    f.write(f"{int(y_true[i])},{probs[i]:.6f}\n")
        logger.info("Wrote %s (%d rows)", raw_path.name, len(y_true))

        pos = int((y_true == 1).sum())
        neg = int((y_true == 0).sum())
        status = {"roc_defined": pos > 0 and neg > 0, "pos": pos, "neg": neg, "n": len(y_true)}
        if not status["roc_defined"]:
            status["reason"] = "single_class_test_set" if neg == 0 or pos == 0 else "insufficient_classes"
            with open(out_dir / f"status_{name}.json", "w") as f:
                json.dump(status, f, indent=2)
            logger.warning("Skipping ROC/PR for %s: degenerate labels pos=%d neg=%d", name, pos, neg)
        else:
            try:
                from sklearn.metrics import precision_recall_curve, roc_curve
                prec, rec, thresh = precision_recall_curve(y_true, probs)
                with open(out_dir / f"pr_curve_{name}.csv", "w") as f:
                    f.write("precision,recall,threshold\n")
                    for i in range(len(prec)):
                        t = thresh[i] if i < len(thresh) else ""
                        f.write(f"{prec[i]},{rec[i]},{t}\n")
                fpr, tpr, _ = roc_curve(y_true, probs)
                with open(out_dir / f"roc_curve_{name}.csv", "w") as f:
                    f.write("fpr,tpr\n")
                    for fp, tp in zip(fpr, tpr):
                        if np.isfinite(tp):
                            f.write(f"{fp},{tp}\n")
                        else:
                            f.write(f"{fp},\n")
                logger.info("Wrote pr_curve_%s.csv and roc_curve_%s.csv (%d PR points)", name, name, len(prec))
            except Exception as e:
                status["error"] = str(e)
                status["roc_defined"] = False
                with open(out_dir / f"status_{name}.json", "w") as f:
                    json.dump(status, f, indent=2)
                logger.warning("PR/ROC curves skip: %s", e)

    # embeddings*.npz -> UMAP/t-SNE -> umap_points.csv
    for emb_name in ("embeddings_entity.npz", "embeddings.npz"):
        emb_path = run_dir / emb_name
        if not emb_path.exists():
            continue
        data = np.load(emb_path, allow_pickle=True)
        emb = data["embeddings"]
        labels = data["labels"] if "labels" in data else np.zeros(emb.shape[0])
        if emb.shape[0] > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(emb.shape[0], 2000, replace=False)
            emb, labels = emb[idx], labels[idx]
        try:
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, random_state=42).fit_transform(emb)
            with open(out_dir / "umap_points.csv", "w") as f:
                f.write("x,y,label\n")
                for i in range(len(labels)):
                    f.write(f"{xy[i,0]},{xy[i,1]},{int(labels[i])}\n")
            logger.info("Wrote umap_points.csv (from %s)", emb_name)
        except Exception as e:
            logger.warning("UMAP/TSNE skip: %s", e)
        break

    # motifs.json -> motifs.csv
    motifs_path = run_dir / "motifs.json"
    if motifs_path.exists():
        with open(motifs_path) as f:
            motifs = json.load(f)
        rows = []
        for group in ("all", "fraud", "non_fraud"):
            m = motifs.get(group, {})
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    rows.append((group, k, v))
        if rows:
            with open(out_dir / "motifs.csv", "w") as f:
                f.write("group,metric,value\n")
                for g, k, v in rows:
                    f.write(f"{g},{k},{v}\n")
            logger.info("Wrote motifs.csv")

    # explanations -> edge list for TikZ/DOT (entity_id, src, dst, src_type, rel, dst_type, weight, hop)
    ex_dir = run_dir / "explanations"
    if ex_dir.is_dir():
        top_path = ex_dir / "top_entities.json"
        if top_path.exists():
            with open(top_path) as f:
                top = json.load(f)
            with open(out_dir / "explanations_edges.csv", "w") as f:
                f.write("entity_id,src,dst,src_type,rel,dst_type,weight,hop\n")
                for rec in top:
                    eid = rec.get("entity_id", "")
                    for edge in rec.get("edges", []):
                        f.write(
                            f"{eid},{edge.get('src','')},{edge.get('dst','')},"
                            f"{edge.get('src_type','entity')},{edge.get('rel','co_occurs')},{edge.get('dst_type','entity')},"
                            f"{edge.get('weight',1)},{edge.get('hop',1)}\n"
                        )
            logger.info("Wrote explanations_edges.csv")

    logger.info("Whitepaper artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
