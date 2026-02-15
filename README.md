# Anchor

Backend and dashboard for an edge voice companion that helps protect elders from fraud. The edge sends **structured event packets** (transcripts, intents, financial events—no raw audio). The backend ingests them, builds a household **Independence Graph**, scores risk (GNN + rules), explains via motifs and subgraphs, and surfaces **risk signals**, **watchlists**, and recommendations. **Read-only for money:** it flags and recommends; it does not execute financial transactions.

```
Edge (batch) → POST /ingest/events → API (FastAPI) + Worker
  → LangGraph pipeline: ingest → normalize → graph_update → Financial Agent
  → risk_score (shared service: HGT or rule fallback) → explain → consent_gate → watchlist → persist
  → Supabase (source of truth) | PyG in-memory (GNN) | Neo4j (optional viz)
  → Next.js dashboard (alerts, protection, agents, graph, replay)
```

| Stack | Role |
|-------|------|
| **Supabase** | Postgres + Auth; sessions, events, entities, risk_signals (fingerprint upsert), watchlists, rings, calibration, agent_runs |
| **FastAPI** | REST + WebSocket `/ws/risk_signals`; routers: households, sessions, alerts, risk_signals, protection, explain, ingest, investigation, agents, outreach, etc. |
| **LangGraph** | Single pipeline: normalize (deterministic) → graph → Financial Agent → risk_score → explain → persist |
| **PyG** | HGT (entity risk + embeddings); GraphGPS/FraudGT for experiments/Elliptic only |
| **Next.js** | Dashboard: auth, protection, alerts (timeline, graph, similar incidents, explain), Run Investigation, agents (catalog, trace), replay |
| **Modal** | Training (HGT, Elliptic); not API/pipeline |

- **Risk:** One place—`domain/risk_scoring_service.py`. Returns calibrated_p, optional rule_score, fusion (0.6×calibrated + 0.4×rule). Conformal bands when calibrated; drift invalidates conformal until recalibration. Rule-only fallback when GNN unavailable.
- **Agents:** Supervisor (INGEST_PIPELINE, NEW_ALERT, NIGHTLY_MAINTENANCE), Financial Security, Graph Drift, Evidence Narrative, Ring Discovery, Calibration, Red-Team, Recurring Contacts, Caregiver Outreach. Status/trace via `GET /agents/status`, `GET /agents/trace`.
- **Graph:** `domain/graph_service.build_graph_from_events`; Independence Graph with MIS-based `independence_violation_ratio` used in rule scoring.

**Quick start**

```bash
pip install -e ".[ml]"   # from repo root
./scripts/run_api.sh     # → http://127.0.0.1:8000
cd apps/web && npm i && npm run dev   # → http://localhost:3000
```

- Pipeline once: `./scripts/run_worker.sh --once --household-id <uuid>`
- Train HGT: `make train` or `make modal-train`
- Test: `make test`

**Docs:** [SETUP.md](SETUP.md) — full setup (Supabase, env, Neo4j). [README_EXTENDED.md](README_EXTENDED.md) — file-by-file reference, schema, event packet, API contracts, agents, tests.

**Repo:** `apps/api/` (FastAPI, pipeline, domain), `apps/worker/` (jobs, persist), `apps/web/` (Next.js), `ml/` (models, graph, train, Modal), `config/` (settings, graph schema), `db/` (bootstrap, migrations 001–024), `scripts/`, `tests/`.

*Python 3.11, FastAPI, Supabase, LangGraph, PyTorch/PyG, Next.js 14, Modal.*
