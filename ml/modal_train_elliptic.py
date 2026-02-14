"""
Modal wrapper for Elliptic (FraudGT-style) training. Run on remote GPU with persisted runs/.

  modal run ml/modal_train_elliptic.py -- --dataset elliptic --model fraud_gt_style --data-dir data/elliptic
  modal run ml/modal_train_elliptic.py -- --dataset elliptic --output runs/elliptic --epochs 100

Script still runs locally: python -m ml.train_elliptic --dataset elliptic --model fraud_gt_style
"""
from __future__ import annotations

import os
import sys

try:
    import modal
except ImportError:
    modal = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if modal is not None:
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2.0",
            "torch-geometric>=2.5.0",
            "scikit-learn",
            "numpy",
            "scipy",
        )
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/anchor",
            copy=True,  # snapshot at build time to avoid "modified during build" errors
            ignore=[
                ".venv", ".git", "__pycache__", "*.pyc", ".next", "node_modules",
                ".ruff_cache", "*.egg-info", ".env", ".env.*", "apps",
            ],
        )
    )

    _runs_volume = modal.Volume.from_name("anchor-runs", create_if_missing=True)
    _app = modal.App("anchor-train-elliptic", image=_image)

    @_app.function(
        volumes={"/root/anchor/runs": _runs_volume},
        timeout=3600,
        # gpu="T4",
    )
    def run_train_elliptic(cli_args: list[str]) -> None:
        """Run ml.train_elliptic.main() with cli_args. Writes to /root/anchor/runs/elliptic (volume)."""
        sys.path.insert(0, "/root/anchor")
        os.chdir("/root/anchor")
        sys.argv = ["ml.train_elliptic", *cli_args]
        try:
            from ml import train_elliptic as elliptic_module
            elliptic_module.main()
        except ImportError as e:
            if "torch_geometric" in str(e) or "torch" in str(e):
                raise RuntimeError(
                    "torch or torch_geometric not available in Modal image."
                ) from e
            raise
        finally:
            _runs_volume.commit()

    @_app.local_entrypoint()
    def main(*args: str) -> None:
        """Pass training args after --, e.g. modal run ml/modal_train_elliptic.py -- --dataset elliptic --model fraud_gt_style --data-dir data/elliptic"""
        if not args:
            args = ("--dataset", "elliptic", "--model", "fraud_gt_style", "--data-dir", "data/elliptic", "--output", "runs/elliptic")
        run_train_elliptic.remote(list(args))


if __name__ == "__main__":
    # Local run without Modal: python -m ml.train_elliptic ... (see docs/modal_training.md)
    pass
