# Anchor test suite

Run all tests (requires Python 3.11+ and project dependencies):

```bash
pip install -r requirements.txt
# Optional ML deps for full coverage:
# pip install torch torch-geometric
pytest tests/ -v
```

## Test layout

| Area | Files | Notes |
|------|--------|------|
| **Config** | `test_config_settings.py`, `test_config_graph.py` | Pipeline/ML settings, graph schema defaults |
| **ML config** | `test_ml_config.py` | YAML loading, path resolution |
| **ML graph** | `test_ml_time_encoding.py`, `test_ml_subgraph.py`, `test_graph_builder.py` | Time encoding, k-hop subgraph, graph builder |
| **ML models** | `test_ml_models.py` | HGTBaseline, GPSRiskModel forward shapes |
| **ML explainers** | `test_ml_motifs.py` | Motif extraction (urgency, bursty contact, etc.) |
| **API state/schemas** | `test_graph_state.py`, `test_schemas.py` | AnchorState, append_log; Pydantic schemas |
| **Pipeline** | `test_pipeline.py`, `test_pipeline_nodes.py` | E2E pipeline, individual nodes (ingest, normalize, risk_score, consent_gate, watchlist, escalation) |
| **Financial agent** | `test_financial_agent.py` | Playbook, consent gating, scam scenarios |
| **API routers** | `test_api_routers.py` | Health, docs, risk_signals list (mocked Supabase) |
| **Broadcast** | `test_broadcast.py` | WebSocket subscriber add/remove |
| **Worker** | `test_worker_jobs.py` | run_graph_builder, run_risk_inference, run_pipeline (mocked) |

Tests that need `torch` / `torch_geometric` use `pytest.importorskip("torch")` (or `torch_geometric`) so they are skipped when ML deps are not installed. API router tests skip when `supabase` is not installed.
