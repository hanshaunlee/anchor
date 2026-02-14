# What the Modal run means & using real data

## What the successful run means

The output you saw:

```
INFO:ml.train:Epoch 10 loss 0.6441
...
INFO:ml.train:Epoch 50 loss 0.6281
INFO:ml.train:Saved to runs/hgt_baseline/best.pt
✓ App completed.
```

means:

- **Training ran to completion** on Modal (remote).
- **Loss went down** (0.64 → 0.63), so the pipeline and model are functioning.
- **Checkpoint was saved** to the Modal Volume at `runs/hgt_baseline/best.pt`.

So the HGT + Modal pipeline is working. Next steps are: use real data, tune hyperparameters, and add eval metrics (e.g. PR-AUC).

---

## Were you using real data?

**No.** For the HGT run you were **not** using real online data:

- `ml/train.py` (HGT) uses **synthetic data only**: `get_synthetic_hetero()` builds a tiny in-memory graph from hardcoded sessions/utterances/entities. The `--data-dir data/synthetic` argument is effectively ignored for that path.
- **Elliptic** is different: `ml/train_elliptic.py` can use the **real Elliptic Bitcoin dataset** from PyG, which downloads when you run with `--dataset elliptic` and `--data-dir data/elliptic`.

So:

- **HGT baseline (Modal)** → synthetic only for now.
- **Elliptic (Modal)** → can use real Elliptic data (online, via PyG).

---

## How to move forward

### 1. Use real data for Elliptic (already supported)

Train on the real Elliptic temporal graph (downloads automatically):

```bash
# Local (downloads to data/elliptic)
python -m ml.train_elliptic --dataset elliptic --model fraud_gt_style --data-dir data

# Modal (same dataset; ensure data is available in the image or mount)
make modal-train-elliptic
# or: modal run ml/modal_train_elliptic.py -- --dataset elliptic --data-dir data/elliptic --epochs 100
```

Note: On Modal, the Elliptic dataset must be present in the container (e.g. download at runtime or mount/cache). The image currently copies the repo; if `data/elliptic` is not in the repo, add a step in the Modal app to download Elliptic inside the container (see below).

### 2. Add real data for HGT (heterogeneous pipeline)

To train HGT on a **real** hetero graph instead of synthetic:

- **HGB (real data):**  
  Use a PyG dataset such as **HGBDataset** (ACM, DBLP, IMDB, Freebase). These are online and return `HeteroData`. You’d add a loader in `ml/train.py` that:
  - Downloads the chosen HGB dataset (or another PyG hetero dataset).
  - Maps its node/edge types and features to the expected schema (or adapt the model to the dataset’s schema), then runs the existing HGT training loop.

- **Option B – Your own data**  
  If you have tables (sessions, utterances, entities, etc.), use the existing `build_hetero_from_tables()` in `ml/graph/builder.py` to build `HeteroData` and save/load it, then point `ml/train.py` at that path so it loads real data instead of calling `get_synthetic_hetero()`.

### 3. Ensure Elliptic downloads on Modal

If Elliptic is not in the image, the Modal app can download it at runtime before training (same as local). For example, in `ml/modal_train_elliptic.py`, ensure the function that runs training first calls the same loader used locally (e.g. `get_elliptic_data(Path("data"))`); PyG will download into that path inside the container. The container has network access, so the download works; later runs can reuse a cached copy if you cache the data directory or use a Modal Volume for `data/elliptic`.

### 4. Suggested order of work

1. **Run Elliptic locally** with real data to confirm download and training.
2. **Run Elliptic on Modal** and, if needed, add a one-time download step so the container has the dataset.
3. **Run HGT on HGB** with `--dataset hgb --hgb-name ACM` (or DBLP/IMDB/Freebase) locally and on Modal.
4. **Add evaluation** (e.g. PR-AUC, recall@k) to both pipelines and log them during training.

---

## Summary

| Pipeline    | Default data   | Real data option                                      |
|------------|----------------|--------------------------------------------------------|
| HGT        | Synthetic      | `--dataset hgb --hgb-name ACM` (or DBLP/IMDB/Freebase) |
| Elliptic   | Synthetic fallback | `--dataset elliptic` (downloads when run)           |

The successful Modal run shows the stack works. Use `--dataset hgb` or `--dataset elliptic` for real online data; then add evaluation metrics and tune.
