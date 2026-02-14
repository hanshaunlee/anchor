# Anchor test suite

Run all tests (requires Python 3.11+ and project dependencies):

```bash
pip install -e ".[ml]"
make test
# or: ruff check apps ml tests && pytest tests apps/api/tests apps/worker/tests ml/tests -v
```

**260+ tests** across config, ML, API, pipeline, worker, scripts, Modal, routers, spec behavior, and strict implementation checks. Test discovery uses `tests/` plus `apps/api/tests`, `apps/worker/tests`, `ml/tests` (pyproject.toml); the main suite is in **`tests/`**.

## Test layout

| Area | Files | Notes |
|------|--------|------|
| **Config** | `test_config_settings.py`, `test_config_graph.py`, `test_config_init.py` | Pipeline/ML settings, graph schema; package exports get_pipeline_settings, get_ml_settings, get_graph_config |
| **ML config** | `test_ml_config.py` | YAML loading, path resolution |
| **ML graph** | `test_ml_time_encoding.py`, `test_ml_subgraph.py`, `test_graph_builder.py`, `test_graph_builder_units.py` | Time encoding, k-hop subgraph, builder (empty events, relationships, hetero, get_mention_list) |
| **ML models** | `test_ml_models.py`, `test_ml_fraud_gt_style.py` | HGTBaseline, GPSRiskModel, FraudGTStyle forward |
| **ML train** | `test_ml_train.py` | focal_loss, _hetero_metadata, get_synthetic_hetero, _get_hetero_labels, train_step |
| **ML inference** | `test_ml_inference.py` | load_model, run_inference (fake checkpoint) |
| **ML cache** | `test_ml_cache_embeddings.py` | EmbeddingCache put/get/matrix/eviction, get_similar_sessions |
| **ML continual** | `test_ml_continual.py` | load_feedback_batch, apply_threshold_adjustment, finetune_last_layer_stub |
| **ML explainers** | `test_ml_motifs.py`, `test_ml_explainers_gnn.py` | Motifs; GNNExplainerStyle, explain_node_gnn_explainer_style |
| **API main** | `test_api_main.py` | root, health, lifespan, WebSocket /ws/risk_signals, app routes |
| **API** | `test_api_deps.py`, `test_api_config.py` | get_supabase (503), require_user (401), Settings |
| **API state/schemas** | `test_graph_state.py`, `test_schemas.py`, `test_schemas_extended.py` | AnchorState, append_log; all Pydantic schemas |
| **Pipeline** | `test_pipeline.py`, `test_pipeline_nodes.py`, `test_pipeline_build_graph.py` | E2E, nodes, build_graph, needs_review, generate_explanations |
| **Financial agent** | `test_financial_agent.py` | Playbook, consent gating, scam scenarios; get_demo_events (copy, structure), playbook with demo events |
| **API routers** | `test_api_routers.py`, `test_routers_agents.py`, `test_routers_extra.py` | Health, risk_signals, ingest; agents (financial/run, status, GET /financial/demo no auth, use_demo_events, GET /financial/trace); households/me, sessions, watchlists, summaries, device |
| **Modal** | `test_modal.py` | modal_train, modal_train_elliptic, modal_app: _REPO_ROOT, app name, run_train / main, hello.remote |
| **Scripts demo** | `test_scripts_demo.py` | run_financial_agent_demo playbook contract, main() prints and exits |
| **Broadcast** | `test_broadcast.py` | WebSocket subscriber add/remove |
| **Worker** | `test_worker_jobs.py`, `test_worker_main.py` | run_graph_builder, run_risk_inference, run_pipeline; main() --once/--household-id, idle path |
| **Scripts** | `test_synthetic_scenarios.py` | make_ts, baseline_normal_events, scam_scenario_events |
| **ML train_elliptic** | `test_ml_train_elliptic.py` | make_synthetic_elliptic, FraudGTStyleSmall, train_epoch, evaluate, get_elliptic_data |
| **ML pg_explainer** | `test_ml_explainers_pg.py` | PGExplainerStyle forward/edge_weights, explain_with_pg |
| **Pipeline financial node** | `test_pipeline_financial_node.py` | financial_security_agent node (empty, with events, consent) |
| **Routers risk_signals** | `test_routers_risk_signals_detail.py` | list with status/severity_min; get_risk_signal, get_similar_incidents, submit_feedback (mocked) |
| **Routers device** | `test_routers_device.py` | POST /device/sync success, 404 device not found, 403 device not in household |
| **Routers ingest** | `test_routers_ingest.py` | POST /ingest/events empty, one event, 403 session not in household |
| **Routers sessions** | `test_routers_sessions_events.py` | list_session_events |
| **Broadcast multi** | `test_broadcast_multi.py` | multiple subscribers, broadcast to all |
| **Config extended** | `test_config_settings_extended.py` | get_ml_config_path with env |
| **Worker extended** | `test_worker_jobs_extended.py` | run_risk_inference cap/shape, run_pipeline outputs, _ml_settings, _pipeline_settings |
| **Spec (behavior)** | `test_pipeline_spec.py`, `test_ml_train_spec.py`, `test_graph_builder_spec.py` | Exact formulas, thresholds, and semantics; fail on regression |
| **Implementation strict** | `test_implementation_strict.py` | Exact behavior and failure paths: ingest preserves prefill, _sessions_from_events, append_log, deps 503/401, focal gamma=0=CE, build_hetero event–mentions, normalize empty/multi-session, financial node on exception, should_review edge cases, watchlist_hit TRIGGERED |
| **GNN product loop** | `test_gnn_product_loop.py` | GNN audit assertions (detection, explanations, watchlists, similar incidents, continual) |
| **Graph policy** | `test_graph_policy.py` | Graph build policy and config |
| **GNN e2e** | `test_gnn_e2e.py` | End-to-end: pipeline with checkpoint, embeddings, similar incidents, model_subgraph (requires checkpoint or `--train`) |

### Specification-based tests

The `*_spec.py` tests are written against **documented behavior**, not just to pass:

- **Pipeline**: Exact risk formula `0.1 + (i % 3) * 0.2`; watchlist/escalation/explanation score thresholds (0.5, 0.6, 0.4); severity `int(1 + score*4)` and threshold 4; consent key typo yields default.
- **ML train**: Focal loss is higher for wrong labels than correct; sum vs mean reduction; near-zero for perfect confidence.
- **Graph builder**: Utterance text from `final_asr` payload; slot `name` → person entity; co-occurrence creates `CO_OCCURS` relationship.

Changing the implementation (e.g. formula or threshold) will break these tests unless the spec is intentionally updated.

### Strict implementation tests

`test_implementation_strict.py` asserts **exact** behavior and **failure paths**. Do not relax assertions—fix the code. These tests are intended to surface bugs (e.g. used-before-defined variables, wrong defaults, missing error handling). If a test fails, fix the implementation rather than the test.

**Bug fixed by strict tests:** `build_hetero_from_tables` used `entity_to_idx` in the Event–MENTIONS–Entity block before it was defined; the strict test would have failed with `NameError`. The code was fixed by defining `entity_to_idx` once before first use.

Tests that need `torch` / `torch_geometric` use `pytest.importorskip(...)`. API router tests skip when `supabase` is not installed.
