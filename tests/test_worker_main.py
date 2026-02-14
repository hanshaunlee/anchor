"""Tests for worker.main: argparse, main() with --once and --household-id."""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ensure_worker_package_path() -> None:
    """Ensure 'worker' resolves to apps/worker (so worker.worker is apps/worker/worker)."""
    for mod in list(sys.modules):
        if mod == "worker" or mod.startswith("worker."):
            del sys.modules[mod]
    apps = str(ROOT / "apps")
    if apps not in sys.path:
        sys.path.insert(0, apps)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "apps" / "api") not in sys.path:
        sys.path.insert(0, str(ROOT / "apps" / "api"))


def test_worker_main_once_household_calls_run_pipeline() -> None:
    """With --once and --household-id, run_pipeline(supabase, household_id) is called."""
    _ensure_worker_package_path()
    with patch.object(sys, "argv", ["worker.main", "--once", "--household-id", "my-hh-99"]):
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=False):
            with patch("worker.worker.jobs.run_pipeline", MagicMock(return_value={})) as mock_run:
                import worker.main
                worker.main.main()
                mock_run.assert_called_once()
                assert mock_run.call_args[0][1] == "my-hh-99"


def test_worker_main_without_once_idle() -> None:
    """Without --once, main() does not call run_pipeline (idle path)."""
    _ensure_worker_package_path()
    with patch.object(sys, "argv", ["worker.main"]):
        with patch("worker.worker.jobs.run_pipeline", MagicMock(return_value={})) as mock_run:
            import worker.main
            worker.main.main()
            mock_run.assert_not_called()


def test_worker_main_supabase_missing_logs_warning() -> None:
    """When SUPABASE_URL or key unset, main still runs (logs warning, supabase=None)."""
    _ensure_worker_package_path()
    with patch.object(sys, "argv", ["worker.main", "--once", "--household-id", "hh1"]):
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=False):
            with patch("worker.worker.jobs.run_pipeline", MagicMock(return_value={})):
                with patch("worker.main.logger") as mock_logger:
                    import worker.main
                    worker.main.main()
                    assert mock_logger.info.called or mock_logger.warning.called
