"""
Reproducible run folder and seed handling for HGT and Elliptic training.
Every run writes to runs/<timestamp>_<gitsha>/ with config, cmd, git, env.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    """Set global seeds and deterministic flags for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _git_sha_dirty() -> tuple[str, bool]:
    """Return (sha_short, dirty). Empty sha if not a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if sha.returncode != 0:
            return "", False
        ref = sha.stdout.strip() or ""
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else False
        return ref, dirty
    except Exception:
        return "", False


def get_default_run_dir() -> Path:
    """runs/<timestamp>_<gitsha>/ (e.g. runs/20250214_120000_abc1234)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sha, dirty = _git_sha_dirty()
    suffix = "_dirty" if dirty else ""
    return Path("runs") / f"{ts}_{sha or 'nogit'}{suffix}"


def setup_run_dir(
    run_dir: Path,
    config_dict: dict[str, Any],
    argv: list[str] | None = None,
) -> None:
    """Create run_dir and write config.yaml, cmd.txt, git.txt, env.txt."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml
    try:
        import yaml

        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    except Exception:
        with open(run_dir / "config.yaml", "w") as f:
            f.write(repr(config_dict))

    # cmd.txt
    with open(run_dir / "cmd.txt", "w") as f:
        f.write(" ".join(f'"{x}"' if " " in x else x for x in (argv or sys.argv)))

    # git.txt
    sha, dirty = _git_sha_dirty()
    with open(run_dir / "git.txt", "w") as f:
        f.write(f"sha={sha}\ndirty={dirty}\n")

    # env.txt
    lines = [
        f"python={sys.version.split()[0]}",
        f"executable={sys.executable}",
    ]
    try:
        import torch

        lines.append(f"torch={getattr(torch, '__version__', '?')}")
    except Exception:
        lines.append("torch=?")
    try:
        import torch_geometric

        lines.append(f"torch_geometric={getattr(torch_geometric, '__version__', '?')}")
    except Exception:
        lines.append("torch_geometric=?")
    try:
        import numpy as np

        lines.append(f"numpy={getattr(np, '__version__', '?')}")
    except Exception:
        lines.append("numpy=?")
    with open(run_dir / "env.txt", "w") as f:
        f.write("\n".join(lines))
