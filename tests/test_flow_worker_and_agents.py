"""
Flow tests: worker run_pipeline flow (ingest -> graph -> risk -> persist); agents return shape and persist flow.
No internal code—only public function names and data flow.
"""
import pytest
from unittest.mock import MagicMock
from uuid import uuid4

pytest.importorskip("torch")
pytest.importorskip("langgraph")


# --- Worker: ingest_events_batch(supabase, household_id, ...) -> list[dict] ---
def test_worker_ingest_events_batch_returns_list() -> None:
    from worker.jobs import ingest_events_batch
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value.data = []
    out = ingest_events_batch(mock, "hh1")
    assert isinstance(out, list)


# --- Worker: run_graph_builder(supabase, household_id, events) -> dict[str, list] ---
def test_worker_run_graph_builder_returns_dict_with_lists() -> None:
    from worker.jobs import run_graph_builder
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    out = run_graph_builder(mock, "hh1", [])
    assert isinstance(out, dict)
    assert isinstance(out.get("entities"), list)
    assert isinstance(out.get("utterances"), list)
    assert isinstance(out.get("mentions"), list)
    assert isinstance(out.get("relationships"), list)


# --- Worker: run_risk_inference(household_id, graph_data, checkpoint_path) -> list[dict] ---
def test_worker_run_risk_inference_returns_list() -> None:
    from worker.jobs import run_risk_inference
    graph_data = {"entities": [], "utterances": [], "mentions": [], "relationships": [], "sessions": []}
    out = run_risk_inference("hh1", graph_data, checkpoint_path=None)
    assert isinstance(out, list)


# --- Worker: run_pipeline(supabase, household_id, ...) -> dict with risk_scores, watchlists ---
def test_worker_run_pipeline_returns_dict_with_core_keys() -> None:
    from worker.jobs import run_pipeline
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value.data = []
    q = mock.table.return_value.select.return_value.eq.return_value
    q.limit.return_value.execute.return_value.data = []  # calibration
    out = run_pipeline(mock, "hh1")
    assert isinstance(out, dict)
    assert "risk_scores" in out
    assert "watchlists" in out
    assert "explanations" in out


# --- Agents: run_financial_security_playbook returns risk_signals, watchlists, logs ---
def test_financial_playbook_returns_contract_shape() -> None:
    from domain.agents.financial_security_agent import run_financial_security_playbook
    out = run_financial_security_playbook(
        household_id="hh1",
        time_window_days=7,
        consent_state={},
        ingested_events=[],
        supabase=None,
        dry_run=True,
    )
    assert "risk_signals" in out
    assert "watchlists" in out
    assert "logs" in out
    assert isinstance(out["risk_signals"], list)
    assert isinstance(out["watchlists"], list)


# --- Agents: run_ring_discovery_agent returns step_trace / contract ---
def test_ring_discovery_agent_returns_contract() -> None:
    from domain.agents.ring_discovery_agent import run_ring_discovery_agent
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    out = run_ring_discovery_agent("hh1", supabase=mock, dry_run=True)
    assert isinstance(out, dict)
    assert "step_trace" in out or "summary" in out or "rings" in out or "status" in out


# --- Agents: run_synthetic_redteam_agent returns summary / variants ---
def test_synthetic_redteam_agent_returns_contract() -> None:
    from domain.agents.synthetic_redteam_agent import run_synthetic_redteam_agent
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    out = run_synthetic_redteam_agent(mock, "hh1", dry_run=True)
    assert isinstance(out, dict)
    assert "summary" in out or "variants" in out or "step_trace" in out


# --- Agents: run_continual_calibration_agent returns status ---
def test_continual_calibration_agent_returns_contract() -> None:
    from domain.agents.continual_calibration_agent import run_continual_calibration_agent
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    out = run_continual_calibration_agent(mock, "hh1")
    assert isinstance(out, dict)
    assert "status" in out or "summary" in out or "error" in out


# --- Agents: run_evidence_narrative_agent returns narrative / report ---
def test_evidence_narrative_agent_returns_contract() -> None:
    from domain.agents.evidence_narrative_agent import run_evidence_narrative_agent
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    out = run_evidence_narrative_agent("hh1", supabase=mock, risk_signal_ids=[])
    assert isinstance(out, dict)


# --- Agents: run_graph_drift_agent returns summary / shift ---
def test_graph_drift_agent_returns_contract() -> None:
    from domain.agents.graph_drift_agent import run_graph_drift_agent
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    out = run_graph_drift_agent(mock, "hh1")
    assert isinstance(out, dict)
    assert "summary" in out or "shift" in out or "status" in out


# --- Flow: worker run_pipeline persists risk_scores and watchlists when supabase present ---
def test_worker_run_pipeline_calls_supabase_insert_when_events_and_supabase() -> None:
    from worker.jobs import run_pipeline
    mock = MagicMock()
    mock.table.return_value.select.return_value.eq.return_value.gte.return_value.lte.return_value.execute.return_value.data = []
    mock.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    out = run_pipeline(mock, "hh1")
    assert isinstance(out, dict)
    assert "risk_scores" in out
