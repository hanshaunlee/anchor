# Modal training (remote GPU)

Run HGT baseline and Elliptic training on Modal with persisted artifacts in a Volume.

## Prerequisites

- Modal account: [modal.com](https://modal.com), then `pip install modal` and `modal setup` (or `modal token new`).
- Repo installed: `pip install -e ".[ml]"` (so `ml.train` and `ml.train_elliptic` are available locally for local runs).
- **Use the project venv** so Modal 1.x is used: `python -m modal run ...` (or run from a shell that has activated `.venv`).

## Environment variables

- **None required for training.** Training scripts do not use Supabase or API keys.
- If you add jobs that need Supabase/API keys, use [Modal Secrets](https://modal.com/docs/guide/secrets): create a secret in the Modal dashboard or `modal secret create anchor-env SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...`, then pass `secrets=[modal.Secret.from_name("anchor-env")]` to `@app.function()`.

## Commands

### HGT baseline (hetero synthetic)

```bash
# Remote (Modal); args after -- are passed to ml.train
modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml
modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml --epochs 100 --device cuda
```

**Artifacts:** written to Modal Volume `anchor-runs` at `runs/hgt_baseline/best.pt`. After the run, pull from the volume or use a one-off command to read the file (see below).

### Elliptic (FraudGT-style)

```bash
# Remote (Modal)
modal run ml/modal_train_elliptic.py -- --dataset elliptic --model fraud_gt_style --data-dir data/elliptic
modal run ml/modal_train_elliptic.py -- --dataset elliptic --output runs/elliptic --epochs 100
```

**Artifacts:** in Volume `anchor-runs` at `runs/elliptic/`:

- `metrics.json`
- `embedding_plot.json` (TSNE 2D)
- `example_explanation_subgraph.json`

### Local execution (no Modal)

Use the underlying scripts so training logic is unchanged:

```bash
# HGT baseline
python -m ml.train --config ml/configs/hgt_baseline.yaml --data-dir data/synthetic

# Elliptic
python -m ml.train_elliptic --dataset elliptic --model fraud_gt_style --data-dir data/elliptic
```

## Makefile

From repo root:

```bash
make modal-train              # HGT baseline on Modal (default args)
make modal-train-elliptic     # Elliptic on Modal (default args)
```

Custom args: run the `modal run ...` commands above manually.

## Persisting artifacts (Volume)

- Both apps use a shared Volume named **`anchor-runs`**.
- The Volume is mounted at `/root/anchor/runs` in the container; training writes to `runs/hgt_baseline` and `runs/elliptic` there.
- After a run, the Volume is committed so files persist across runs.
- To download artifacts to your machine, use the Modal CLI or a small one-off function that reads from the volume and returns the file (or use a Modal Network File System if you prefer).

## GPU

To use a GPU, uncomment the `gpu="T4"` (or `gpu="A10G"`) line in:

- `ml/modal_train.py` inside `@_app.function(...)`
- `ml/modal_train_elliptic.py` inside `@_app.function(...)`

## Troubleshooting

- **`torch_geometric` or `torch` not found in container:** The image installs `torch` and `torch-geometric` via pip. If you see a clear error about a missing native extension (e.g. `torch-scatter`), add it to the image with `.pip_install("torch-scatter", ...)` and a compatible PyTorch version, or run without scatter (the current image does not install `torch-scatter`/`torch-sparse` to keep the image small).
- **Runs not persisting:** Ensure the function calls `_runs_volume.commit()` in a `finally` block (already present). If the run crashes before commit, the volume may not update.
