"""
Modal wrapper for HGT baseline training. Run on remote GPU/CPU with persisted runs/.

  modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml
  modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml --epochs 100 --device cuda

Script still runs locally: python -m ml.train --config ml/configs/hgt_baseline.yaml
"""
from __future__ import annotations

import os
import sys

# Optional: only import Modal when actually running remotely
try:
    import modal
except ImportError:
    modal = None

# Repo root when running locally (for python -m ml.train)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if modal is not None:
    # Image: Python 3.11 + torch + torch-geometric; repo mounted at /root/anchor (Modal 1.x: use Image.add_local_dir)
    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2.0",
            "torch-geometric>=2.5.0",
            "scikit-learn",
            "numpy",
            "scipy",
            "pyyaml",
        )
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/anchor",
            copy=True,  # snapshot at build time to avoid "modified during build" errors
            ignore=[
                ".venv", ".git", "__pycache__", "*.pyc", ".next", "node_modules",
                ".ruff_cache", "*.egg-info", ".env", ".env.*", "apps",
                "runs",  # avoid non-empty path so Volume can mount at /root/anchor/runs
            ],
        )
    )

    _runs_volume = modal.Volume.from_name("anchor-runs", create_if_missing=True)
    _app = modal.App("anchor-train", image=_image)

    @_app.function(
        volumes={"/root/anchor/runs": _runs_volume},
        timeout=3600,
        # Uncomment for GPU:
        # gpu="T4",
    )
    def run_train(cli_args: list[str]) -> None:
        """Run ml.train.main() with sys.argv set to cli_args. Artifacts go to /root/anchor/runs (volume)."""
        sys.path.insert(0, "/root/anchor")
        os.chdir("/root/anchor")
        sys.argv = ["ml.train", *cli_args]
        try:
            from ml import train as train_module
            train_module.main()
        except ImportError as e:
            if "torch_geometric" in str(e) or "torch" in str(e):
                raise RuntimeError(
                    "torch or torch_geometric not available in Modal image. "
                    "Check that the image installs PyTorch and PyG."
                ) from e
            raise
        finally:
            _runs_volume.commit()

    @_app.function(
        volumes={"/root/anchor/runs": _runs_volume},
        timeout=60,
    )
    def read_checkpoint(remote_path: str = "runs/hgt_baseline/best.pt") -> bytes:
        """Read checkpoint bytes from the runs volume (for downloading to local). Path relative to repo root, e.g. runs/hgt_baseline/best.pt."""
        # Volume is mounted at /root/anchor/runs
        path = remote_path.replace("\\", "/").strip("/")
        if not path.startswith("runs/"):
            path = "runs/" + path
        full = f"/root/anchor/{path}"
        with open(full, "rb") as f:
            return f.read()

    @_app.local_entrypoint()
    def main(*args: str) -> None:
        """Pass training args after --, e.g. modal run ml/modal_train.py::main -- --config ml/configs/hgt_baseline.yaml --data-dir data/synthetic"""
        if not args:
            args = ("--config", "ml/configs/hgt_baseline.yaml", "--data-dir", "data/synthetic")
        run_train.remote(list(args))

    @_app.local_entrypoint()
    def download_checkpoint(
        remote_path: str = "runs/hgt_baseline/best.pt",
        local_path: str = "runs/hgt_baseline/best.pt",
    ) -> None:
        """Download checkpoint from Modal volume to local file.
        Example: modal run ml/modal_train.py::download_checkpoint --local-path runs/hgt_baseline/best.pt
        """
        import os
        data = read_checkpoint.remote(remote_path)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        print(f"Downloaded {len(data)} bytes to {local_path}")


if __name__ == "__main__":
    # Local run without Modal: python -m ml.train --config ... (see README_EXTENDED.md § 7)
    pass
