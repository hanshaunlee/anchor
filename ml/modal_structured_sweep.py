"""
Modal pipeline: generate large structured synthetic datasets (2k–5k nodes) and train HGT over 5 seeds.
Reports mean ± std ROC-AUC and PR-AUC for statistical rigor.

  modal run ml/modal_structured_sweep.py::main --n-entities 2500 --epochs 100
  modal run ml/modal_structured_sweep.py::main --n-entities 5000 --epochs 150 --gpu

Where to access results (Modal Volume → local):
  1. Summary (mean ± std ROC/PR-AUC): modal run ml/modal_structured_sweep.py::download_summary
  2. Full sweep dirs (seed_0..seed_4 with metrics, checkpoints): modal volume get anchor-runs structured_sweep_5seeds ./runs/structured_sweep_5seeds
     (Volume is mounted at runs/, so the path inside the volume is "structured_sweep_5seeds", not "runs/structured_sweep_5seeds".)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SWEEP_DIR = "runs/structured_sweep_5seeds"
N_SEEDS = 5


if modal is not None:
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2.0",
            "torch-geometric>=2.5.0",
            "scikit-learn",
            "numpy",
            "scipy",
            "pyyaml",
            "networkx",
        )
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/anchor",
            copy=True,
            ignore=[
                ".venv", ".git", "__pycache__", "*.pyc", ".next", "node_modules",
                ".ruff_cache", "*.egg-info", ".env", ".env.*", "apps",
                "runs",
            ],
        )
    )

    _runs_volume = modal.Volume.from_name("anchor-runs", create_if_missing=True)
    _app = modal.App("anchor-structured-sweep", image=_image)

    def _sweep_impl(
        n_entities: int,
        epochs: int,
        use_gpu: bool = False,
    ) -> dict:
        """Run 5 seeds: for each, generate structured_synthetic with n_entities and train. Then aggregate mean ± std."""
        sys.path.insert(0, "/root/anchor")
        os.chdir("/root/anchor")
        os.environ["ANCHOR_STRUCTURED_N_ENTITIES"] = str(n_entities)
        n_sessions = min(400, max(180, n_entities // 3))
        os.environ["ANCHOR_STRUCTURED_N_SESSIONS"] = str(n_sessions)

        results = []
        for seed in range(N_SEEDS):
            run_dir = f"{SWEEP_DIR}/seed_{seed}"
            argv = [
                "--config", "ml/configs/hgt_baseline.yaml",
                "--dataset", "structured_synthetic",
                "--run-dir", run_dir,
                "--seed", str(seed),
                "--epochs", str(epochs),
            ]
            if use_gpu:
                argv.extend(["--device", "cuda"])
            sys.argv = ["ml.train", *argv]
            try:
                from ml import train as train_module
                train_module.main()
            except Exception as e:
                results.append({"seed": seed, "error": str(e), "run_dir": run_dir})
                continue
            metrics_path = f"/root/anchor/{run_dir}/metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                results.append({
                    "seed": seed,
                    "run_dir": run_dir,
                    "test_roc_auc": m.get("test_roc_auc"),
                    "test_pr_auc": m.get("test_pr_auc"),
                    "val_roc_auc": m.get("val_roc_auc"),
                    "val_pr_auc": m.get("val_pr_auc"),
                })
            else:
                results.append({"seed": seed, "run_dir": run_dir, "error": "metrics.json not found"})

        # Aggregate
        roc = [r["test_roc_auc"] for r in results if r.get("test_roc_auc") is not None]
        pr = [r["test_pr_auc"] for r in results if r.get("test_pr_auc") is not None]
        import numpy as np
        summary = {
            "n_entities": n_entities,
            "n_sessions": n_sessions,
            "epochs": epochs,
            "n_seeds": N_SEEDS,
            "per_seed": results,
            "test_roc_auc_mean": float(np.mean(roc)) if roc else None,
            "test_roc_auc_std": float(np.std(roc)) if len(roc) > 1 else None,
            "test_pr_auc_mean": float(np.mean(pr)) if pr else None,
            "test_pr_auc_std": float(np.std(pr)) if len(pr) > 1 else None,
            "val_roc_auc_mean": float(np.mean([r["val_roc_auc"] for r in results if r.get("val_roc_auc") is not None])) if results else None,
            "val_pr_auc_mean": float(np.mean([r["val_pr_auc"] for r in results if r.get("val_pr_auc") is not None])) if results else None,
        }
        summary_path = f"/root/anchor/{SWEEP_DIR}/summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        _runs_volume.commit()
        return summary

    @_app.function(volumes={"/root/anchor/runs": _runs_volume}, timeout=7200)
    def run_sweep(n_entities: int = 2500, epochs: int = 100) -> dict:
        """CPU: 5 seeds, structured_synthetic with n_entities."""
        return _sweep_impl(n_entities, epochs, use_gpu=False)

    @_app.function(volumes={"/root/anchor/runs": _runs_volume}, timeout=7200, gpu="T4")
    def run_sweep_gpu(n_entities: int = 2500, epochs: int = 100) -> dict:
        """GPU (T4): 5 seeds, structured_synthetic with n_entities."""
        return _sweep_impl(n_entities, epochs, use_gpu=True)

    @_app.local_entrypoint()
    def main(
        n_entities: int = 2500,
        epochs: int = 100,
        gpu: bool = False,
    ) -> None:
        """Run 5-seed sweep on Modal. Example: modal run ml/modal_structured_sweep.py -- --n-entities 2500 --epochs 100 [--gpu]"""
        if gpu:
            summary = run_sweep_gpu.remote(n_entities=n_entities, epochs=epochs)
        else:
            summary = run_sweep.remote(n_entities=n_entities, epochs=epochs)
        print("Sweep summary:")
        print(json.dumps(summary, indent=2))

    @_app.function(volumes={"/root/anchor/runs": _runs_volume}, timeout=30)
    def read_summary() -> str:
        with open(f"/root/anchor/{SWEEP_DIR}/summary.json") as f:
            return f.read()

    @_app.local_entrypoint()
    def download_summary(local_path: str = "runs/structured_sweep_5seeds/summary.json") -> None:
        """Download summary.json from Modal volume to local."""
        content = read_summary.remote()
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"Downloaded to {path}")


if __name__ == "__main__":
    pass
