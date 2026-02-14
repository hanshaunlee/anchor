"""Tests for scripts/run_financial_agent_demo: entrypoint and output contract."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root and apps/api on path (conftest does this for tests, but we import script logic)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(ROOT / "apps" / "api"))

from api.agents.financial_agent import get_demo_events, run_financial_security_playbook


def test_demo_script_playbook_contract() -> None:
    """What the script runs: playbook with get_demo_events(), consent, dry_run. Result has logs, motif_tags, risk_signals, watchlists."""
    events = get_demo_events()
    result = run_financial_security_playbook(
        household_id="hh1",
        time_window_days=7,
        consent_state={"share_with_caregiver": True, "watchlist_ok": True},
        ingested_events=events,
        supabase=None,
        dry_run=True,
    )
    assert "logs" in result
    assert "motif_tags" in result
    assert "risk_signals" in result
    assert "watchlists" in result
    assert isinstance(result["logs"], list)
    assert isinstance(result["risk_signals"], list)
    assert len(result["risk_signals"]) >= 1


def test_demo_script_main_prints_and_exits_zero() -> None:
    """Running main() (script entrypoint) prints and exits 0; skip if script file is missing."""
    script_path = ROOT / "scripts" / "run_financial_agent_demo.py"
    if not script_path.exists():
        pytest.skip("scripts/run_financial_agent_demo.py not found")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_financial_agent_demo",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with patch("sys.stdout") as mock_stdout:
        mod.main()
    written = "".join(str(call[0][0]) for call in mock_stdout.write.call_args_list if call[0])
    assert "Financial Security Agent" in written or "Running" in written
    assert "risk" in written.lower() or "Risk" in written
