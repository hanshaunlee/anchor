# Whitepaper Checklist: What You Need for the Research Paper

Use one **strong run** (e.g. **runs/hgt_structured_02** or a sweep seed like **runs/structured_sweep_5seeds/seed_0**). Paths below are relative to the repo root.

---

## 1. Export plot-ready artifacts (one command)

From repo root:

```bash
python ml/export_whitepaper_artifacts.py --run-dir runs/hgt_structured_02
```

(Replace `runs/hgt_structured_02` with your run dir, e.g. `runs/structured_sweep_5seeds/structured_sweep_5seeds/seed_0` if you downloaded the Modal sweep into the nested path.)

This writes **`<run_dir>/whitepaper/`** with:

| For the paper | File | Use |
|---------------|------|-----|
| **Training curve** | `loss.csv` | `epoch,train_loss` — learning curve plot |
| **Per-epoch metrics** | `metrics_per_epoch.csv` | `epoch,train_loss,val_pr_auc,val_roc_auc,test_pr_auc,test_roc_auc` — val/test curves |
| **PR curve (test)** | `pr_curve_test.csv` | `precision,recall,threshold` — PR curve plot |
| **ROC curve (test)** | `roc_curve_test.csv` | `fpr,tpr` — ROC curve plot |
| **Raw scores** | `y_true_y_score_test.csv`, `y_true_y_score_val.csv` | Threshold tables, calibration curve, score histograms (fraud vs non-fraud) |
| **Embedding viz** | `umap_points.csv` | `x,y,label` — 2D UMAP/t-SNE plot |
| **Motifs** | `motifs.csv` | Structural motif counts (for tables) |
| **Explanations** | `explanations_edges.csv` | Edge list for explainer subgraphs (optional; can be large) |

---

## 2. Structural significance (Cohen's d, t-tests, degree)

```bash
python scripts/analyze_structural_significance.py <run_dir>
python scripts/analyze_structural_significance.py <run_dir> -o <run_dir>/structural_report.json
```

Gives you:

- **Cohen's d** and **t-test p-values** for: cross_session_ratio, triangle_count, k_core_number, betweenness, device_reuse_count, temporal_burst_score  
- **Degree distribution:** mean degree (fraud vs normal), degree Cohen's d, degree t-test p-value, **Kolmogorov–Smirnov** (statistic + p-value)  
- **Motif/structure means:** triangles, cross-session ratio, modularity, k-core (fraud vs normal)  
- **Model metrics** (if `metrics.json` exists): test/val ROC-AUC, PR-AUC  

Use for: **Methods** (structural signal), **Results** (effect sizes, significance), **Supplementary** (full stats).

---

## 3. Aggregate metrics and multi-seed (if you ran the sweep)

- **`runs/structured_sweep_5seeds/summary.json`**  
  - `test_roc_auc_mean`, `test_roc_auc_std`, `test_pr_auc_mean`, `test_pr_auc_std`  
  - `per_seed`: list of per-seed test/val ROC-AUC and PR-AUC  

Use for: **Results** — report “Test ROC-AUC = 0.XX ± 0.YY (5 seeds)” and same for PR-AUC.

---

## 4. Run-level JSON (for tables / reproducibility)

From the **run dir** (not whitepaper/):

| File | Use |
|------|-----|
| `metrics.json` | Final test/val ROC-AUC, PR-AUC, accuracy — main results table |
| `graph_stats.json` | Node/edge counts, label balance, `structured_stats` (if structured run) — dataset table |
| `data_structured_synthetic.json` | Same structural stats + prevalence — Methods/Supplementary |

---

## 5. LaTeX / figures

- **Curves:** Use `pr_curve_test.csv`, `roc_curve_test.csv`, `loss.csv`, `metrics_per_epoch.csv` in pgfplots or your plotting script.  
- **UMAP:** Use `umap_points.csv` (x, y, label) for the embedding figure.  
- **Sanity-check:** Curve-derived AUCs from the CSVs should match `metrics.json` (test_roc_auc, test_pr_auc).

---

## 6. Suggested “Results” sentence

For a **single run** (e.g. hgt_structured_02):

- “Test ROC-AUC 0.93, test PR-AUC 0.78 (positive rate 27%).”

For the **5-seed sweep**:

- “Test ROC-AUC 0.XX ± 0.YY, test PR-AUC 0.XX ± 0.YY (5 seeds, N = 2500 entities, 100 epochs).”

---

## Quick reference: one strong run

1. Pick run: e.g. **runs/hgt_structured_02** or **runs/structured_sweep_5seeds/structured_sweep_5seeds/seed_0**.  
2. Export whitepaper: `python ml/export_whitepaper_artifacts.py --run-dir <run_dir>`.  
3. Structural report: `python scripts/analyze_structural_significance.py <run_dir> -o <run_dir>/structural_report.json`.  
4. Use **`<run_dir>/whitepaper/`** for plots and **`<run_dir>/structural_report.json`** + **`metrics.json`** + **`summary.json`** (if sweep) for numbers and tables.
