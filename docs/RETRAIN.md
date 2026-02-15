# How to retrain the GNN with the new config (128-D + retrieval head)

The config and model now use **128-D hidden**, a **retrieval projection head** (`embed_dim: 128`), and optional **contrastive loss**. Follow these steps to produce a checkpoint that writes 128-D embeddings and uses the v2 similarity RPC.

## 1. Prerequisites

- **DB**: Run migration **022** so `risk_signal_embeddings.embedding_vector_v2 vector(128)` and the RPC `similar_incidents_by_vector_v2` exist.
- **Config**: `ml/configs/hgt_baseline.yaml` already has:
  - `model.hidden_channels: 128`
  - `model.embed_dim: 128`
  - `train.contrastive_weight: 0.0` (set to e.g. `0.2` to train the retrieval head with same/different-label pairs)

## 2. Local training (no Modal)

From repo root:

```bash
# Default: synthetic graph, config from ml/configs/hgt_baseline.yaml, output runs/hgt_baseline/best.pt
python -m ml.train

# Explicit config and output
python -m ml.train --config ml/configs/hgt_baseline.yaml --output runs/hgt_baseline

# Override epochs / LR
python -m ml.train --config ml/configs/hgt_baseline.yaml --epochs 100 --lr 0.0005

# Optional: train on HGB dataset (different node types; match checkpoint to your inference graph)
python -m ml.train --config ml/configs/hgt_baseline.yaml --dataset hgb --hgb-name ACM --data-dir data/hgb
```

Checkpoint is written to `runs/hgt_baseline/best.pt` (or `--output` path). It includes `embed_dim: 128` and `embed_proj` state.

## 3. Training on Modal (remote GPU)

From repo root:

```bash
# Default args (config + data-dir)
modal run ml/modal_train.py::main

# With explicit config and epochs
modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml --epochs 80

# Optional: use GPU (uncomment gpu="T4" in modal_train.py first)
modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml --device cuda --epochs 100
```

Checkpoints are stored on the Modal volume `anchor-runs` at `/root/anchor/runs/`. Download to local:

```bash
modal run ml/modal_train.py::download_checkpoint --remote-path runs/hgt_baseline/best.pt --local-path runs/hgt_baseline/best.pt
```

## 4. Point the app at the new checkpoint

- Set **ANCHOR_ML_CHECKPOINT_PATH** (or your app’s config) to the path of `best.pt`, e.g. `runs/hgt_baseline/best.pt`.
- Ensure **ANCHOR_ML_*** embedding_dim is **128** (default in `config/settings.py` is already 128).

## 5. After retraining

- **Inference** will produce 128-D vectors from the **retrieval head** (`embed_proj`) when the checkpoint contains it; otherwise it falls back to the classifier hidden (also 128-D with the new config).
- **Worker / Financial agent** write to `embedding_vector_v2` when `len(vec) == 128` (migration 022).
- **Similar incidents** use the RPC `similar_incidents_by_vector_v2` when the query row has `dim == 128`.

## 6. Optional: contrastive retrieval objective

To train the retrieval head for better similar-incidents / rings / watchlists:

1. In `ml/configs/hgt_baseline.yaml`, set:
   - `train.contrastive_weight: 0.2` (or 0.1–0.5).
2. Retrain as above. The training step will add an in-batch contrastive term: same-label pairs pulled together, different-label pairs pushed apart (on the projected embedding).

Synthetic data has minimal labels; the effect is stronger with real or redteam data where same risk pattern / motif tag can be used as positives.

## 7. Model registry (Phase 4)

Risk scoring can use a **model registry** so you can switch or add models via config:

- Set **ANCHOR_GNN_MODEL** to the model name (default is `hgt`).
- Supported: `hgt` or `hgt_baseline` → HGT baseline (risk + embeddings). `fraudgt_entity` → stub (not implemented for pipeline yet).
- When unset or `hgt`, the service uses the same HGT path as before; when set, the registry returns a runner that loads the checkpoint and runs inference. This allows adding a second model (e.g. FraudGT for entity retrieval) later without changing callers.

## 8. Quick reference

| Goal                         | Command / change |
|-----------------------------|------------------|
| Local train, default config | `python -m ml.train` |
| Modal train                 | `modal run ml/modal_train.py::main` |
| Download Modal checkpoint   | `modal run ml/modal_train.py::download_checkpoint --local-path runs/hgt_baseline/best.pt` |
| Enable contrastive          | Set `train.contrastive_weight: 0.2` in YAML and retrain |
| Use new checkpoint in app   | Set `ANCHOR_ML_CHECKPOINT_PATH` to `best.pt` path |
