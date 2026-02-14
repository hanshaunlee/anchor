# Anchor demo moments (hackathon)

## 1. Temporal: “Watch the risk rise as the scammer tests with small interactions.”

- **TGAT-style time encoding**: Node and edge features include sinusoidal time encoding (`ml/graph/time_encoding.py`). `build_hetero_from_tables` attaches timestamps to utterances/entities/sessions and concatenates time encoding to node features and to `co_occurs` edge_attr.
- **time_to_flag**: Pipeline state has `time_to_flag` (seconds from first event to first score above threshold). Replay a sequence of events and show risk increasing before the “big bad event.”
- **Plot**: Compare detection latency (time_to_flag) vs static baseline in eval script or notebook.

## 2. Subgraph-local: “We only compute where the evidence is — scalable, privacy-aware.”

- **k-hop extraction**: `ml/graph/subgraph.py` — `extract_k_hop(heterodata, seed_nodes_by_type, k, time_window)` returns induced subgraph + node maps.
- **Touched entities**: `get_touched_entities(new_mentions, new_relationships, entity_id_to_index)` for each new batch.
- **Caching**: `ml/cache/embeddings.py` — `EmbeddingCache` stores (node_id/session_id, embedding, ts); reuse for retrieval and explainers.
- **Dashboard**: “Subgraph evidence” is exactly the k-hop subgraph that was scored.

## 3. Similar Incidents: “Not just alerting — it gives precedent.”

- **API**: `GET /risk_signals/{id}/similar?top_k=5` — cosine nearest neighbors from `risk_signal_embeddings`.
- **Table**: `risk_signal_embeddings` (risk_signal_id, household_id, embedding JSONB). Worker writes embedding when persisting a risk signal.
- **UI**: Similar Incidents panel: 3–5 most similar past incidents and outcomes (confirmed scam vs false positive).

## 4. Two-layer explanation: “The model agrees with the motif; here’s the evidence.”

- **Layer A**: `ml/explainers/motifs.py` — rule-based motifs: new contact + urgency, bursty contact, device switch, contact→sensitive intent cascade.
- **Layer B**: Model-based (PGExplainer) minimal evidence subgraph with ranked edges.
- **explanation_json**: `motif_tags`, `model_subgraph`, `timeline_snippet` (3–6 key events). Each alert shows plain-English motif tags and clickable subgraph.

## 5. HITL: “The system learns household-specific behavior safely.”

- **needs_review branch**: LangGraph conditional after `consent_gate` — if severity >= 4 and `consent.share_with_caregiver` → `needs_review` node.
- **Feedback**: `POST /risk_signals/{id}/feedback` (Confirm scam / False alarm / Unsure). False positive → `household_calibration.severity_threshold_adjust` += 0.1.
- **Stub**: `ml/continual/finetune_last_layer.py` — fine-tuning queue (weekly) hook.

## 6. Edge device: “Closed-loop improvement.”

- **Event type**: `watchlist_hit` — payload can include `entity_id` / `matched_entity_id`, `watchlist_id`.
- **Graph**: `GraphBuilder.process_events` adds Entity —[TRIGGERED]→ evidence (relationship with `rel_type` TRIGGERED, evidence list).
- **Device sync**: `POST /device/sync` returns watchlist delta + version; device can send `watchlist_hit` when a pattern matches.

## 7. Elliptic: “This model class is validated on a canonical benchmark.”

- **Script**: `python -m ml.train_elliptic --dataset elliptic --model fraud_gt_style`
- **Output**: `runs/elliptic/metrics.json` (PR-AUC, accuracy), `embedding_plot.json` (TSNE/UMAP), `example_explanation_subgraph.json`.
- **Fallback**: If Elliptic not downloaded, uses synthetic graph; same interface.

## 8. Real action: "The system can notify the caregiver — with consent and evidence."

- **Caregiver Outreach Agent**: Ten-step playbook: load incident + consent, policy gate, choose channel (respecting quiet hours), evidence bundle, generate caregiver + elder-safe message, create `outbound_actions` row, dispatch via provider (SMS/email/voice), update risk_signal to escalated, broadcast to UI.
- **Consent**: `consent_allow_outbound_contact` must be true (default opt-in false). When false, outreach is **suppressed** and a row is still written with `status=suppressed` for audit.
- **Roles**: Only **caregiver** or **admin** can trigger outreach (elder gets 403). Elder can see only `elder_safe_message` for outbound actions in their household.
- **Demo**: `PYTHONPATH=".:apps/api" python scripts/run_outreach_demo.py` — runs the agent with a mock signal and MockProvider (logs only; no real send). With Supabase and migrations 011/012, use **Alerts → [signal] → Notify caregiver** (preview → confirm send) or `POST /actions/outreach` / `POST /agents/outreach/run`.
- **Worker**: `run_outreach_for_new_signals(supabase, household_id)` finds open signals with severity ≥ threshold and consent, and runs outreach (enqueue pattern; call from cron or after pipeline).

---

## Real flows (post-cohesion)

These flows use live backend artifacts; fixtures are used only in demo mode or when the API is unavailable.

- **Similar incidents**: `GET /risk_signals/{id}/similar` returns `available: false` when no GNN embedding was persisted (e.g. model not run). UI shows “Unavailable” instead of synthetic results. Financial agent and worker persist embeddings only when the model returns them.
- **Centroid watchlists**: Pipeline creates `embedding_centroid` watchlists with `source.risk_signal_ids` and `source.window`. Worker matches new embeddings to centroid watchlists and creates `watchlist_embedding_match` risk signals. Alert detail shows “Matched centroid watchlist.”
- **Model evidence subgraph**: `model_subgraph` is populated from PGExplainer when the model runs. `POST /risk_signals/{id}/explain/deep_dive?mode=pg` persists `deep_dive_subgraph`. UI graph card has PGExplainer / Deep dive toggle and “Compute deep dive” when not yet run.
- **Replay**: “Refresh from API” loads risk_signals and explanation subgraphs from the Financial demo or run; step_trace from the agent when present. Fixture mode only when API fails or demo mode is on.
- **Agents**: Graph Drift shows “Copy retrain command” when drift is detected. Evidence Narrative persists `narrative_reports` and “View report” links to `/reports/narrative/[id]`. Ring Discovery has Rings page and “View ring” from alert. Calibration and Red-Team have report pages and “View calibration report” / “View red-team report”; Red-Team “Open in replay” loads trace/timeline from last run.
- **Graph build**: Single place `domain.graph_service.build_graph_from_events(household_id, events, supabase=…)`; pipeline, graph router, and worker use it. Optional persist when supabase is provided.
