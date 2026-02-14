# Anchor Agents

## Financial Security Agent

The **Financial Security Agent** runs an ordered “financial protection playbook” to protect an elder user. It is **read-only** in terms of external actions: it can recommend, draft, and flag; it **cannot move money** or execute financial transactions.

### Ordered playbook (tasks)

1. **Ingest & normalize**
   - Pull recent events for the household in a configurable time window (default 7 days) from Supabase, or use pre-filled events from the LangGraph state.
   - Normalize to a consistent internal representation using the existing GraphBuilder: utterances, intents, entities, mentions, relationships.

2. **Detect risk patterns (rule + model)**
   - **Motif/rule layer**: scam-like patterns (e.g. “urgency + sensitive info request”, “new unknown contact”, “repeat attempts”, “new payee attempt” when available).
   - **Model layer** (optional): GNN risk scoring (HGT/GraphGPS/FraudGT-style) on entity subgraphs when a checkpoint is available.
   - Output: calibrated risk score, severity (1–5), and uncertainty.

3. **Investigation package**
   - Evidence bundle for the UI:
     - `motif_tags`
     - `timeline_snippet` (3–6 key events)
     - Evidence subgraph (nodes/edges with importance)
     - “What changed vs baseline” summary (e.g. new contact first seen, spike in attempts).
   - Uses existing explainers (motifs + PGExplainer/GNNExplainer) when available.

4. **Protective recommendations (non-destructive)**
   - `recommended_action` JSON with a tailored checklist, e.g.:
     - “Call back using saved contact”
     - “Do not share OTP/codes”
     - “Freeze unknown caller interactions for 60 minutes” (recommendation only)
     - “Enable bank alerts”
     - “Change passwords / 2FA”
     - “Verify payee”
     - “Review recent transactions and confirm unknown merchants” (when applicable)

5. **Watchlist synthesis**
   - Converts risk into compact watchlist items for the edge device: phone/email/name hashes, risky topic keywords.
   - Inserts into the `watchlists` table with priority and expiry (when not in dry-run and consent allows).

6. **Consent-gated escalation draft**
   - If severity ≥ threshold **and** `consent_state.share_with_caregiver === true`: draft an escalation message (stored in `recommended_action` or explanation; **not sent**).
   - Otherwise: no escalation payload is persisted beyond the elder’s allowed scope. If model confidence is low, the agent recommends a clarification question instead of escalating.

7. **Persist + notify**
   - Inserts/updates `risk_signals` with:
     - `signal_type` (e.g. `possible_scam_contact`, `social_engineering_risk`, `payment_anomaly`)
     - `severity` (1–5), `score`, `explanation` (motifs, subgraph, timeline), `recommended_action`, `status = open`
   - Broadcasts new alerts over WebSocket `/ws/risk_signals` when run via the API with persist enabled.

### Integration

- **LangGraph**: The node `financial_security_agent` runs after `graph_update` and before `risk_score`. It uses state (utterances, entities, mentions, relationships); when run inside the pipeline it does **not** persist to the DB (no Supabase in context). Use **POST /agents/financial/run** to run with persist and broadcast.
- **On-demand API**: **POST /agents/financial/run** with optional body `{ household_id?, time_window_days?, dry_run? }`. If `dry_run=true`, no DB write; response includes the computed risk_signals and watchlists for preview.
- **Consent**: Respects `sessions.consent_state` (e.g. `share_with_caregiver`, `watchlist_ok`). Does not expose redacted text when consent disallows.

### API usage

- **Run agent (persist or preview)**  
  `POST /agents/financial/run`  
  Body: `{ "time_window_days": 7, "dry_run": false }`  
  Returns: `run_id`, `risk_signals_count`, `watchlists_count`, `inserted_signal_ids`, `logs`; when `dry_run=true` also `risk_signals` and `watchlists`.

- **Agent status**  
  `GET /agents/status`  
  Returns: list of agents with `agent_name`, `last_run_at`, `last_run_status`, `last_run_summary`.

- **Financial run trace**  
  `GET /agents/financial/trace?run_id=<uuid>`  
  Returns: one row from `agent_runs` for that financial run.

### Testing the Financial Security Agent

**1. Demo via API (no auth) — see input + output**

```bash
curl -s http://127.0.0.1:8000/agents/financial/demo | jq
```

Returns `input_events` (3 demo events: Medicare urgency + share_ssn + phone) and `output` (logs, motif_tags, risk_signals, watchlists). No DB write.

**2. Via API with auth (optional demo events)**

- **Preview with demo events (no DB write, full input/output in response):**  
  `POST /agents/financial/run`  
  Body: `{ "dry_run": true, "use_demo_events": true }`  
  Header: `Authorization: Bearer <your_supabase_jwt>`  
  Response includes `input_events` and `risk_signals`, `watchlists`, `motif_tags`, `timeline_snippet`.
- **Live run on DB events:**  
  Body: `{ "dry_run": false }` — agent fetches events for your household from the last 7 days.
- **Live run on demo events (persist demo output to your household):**  
  Body: `{ "dry_run": false, "use_demo_events": true }`
- **Check status:**  
  `GET /agents/status`  
  `GET /agents/financial/trace?run_id=<uuid>`

**3. Unit tests (no API, no DB)**

```bash
pytest tests/test_financial_agent.py -v
```

**4. One-command demo harness (recommended for judges)**

```bash
PYTHONPATH=".:apps/api" python3 scripts/demo_replay.py
```

Writes `demo_out/risk_chart.json`, `explanation_subgraph.json`, `agent_trace.json`, and `scenario_replay.json`. Use `--ui` to copy the replay payload into the web app fixtures; use `--launch-ui` to also start the Next.js dev server in demo mode.

**5. Local script (no API, playbook only)**

```bash
PYTHONPATH=".:apps/api" python3 scripts/run_financial_agent_demo.py
```

**6. As part of the LangGraph pipeline**

The pipeline runs the financial agent as a node after `graph_update`. It only updates in-memory state (no persist). To persist financial outputs, run the agent via **POST /agents/financial/run** (e.g. after ingest or on a schedule).

### Safety / policy

- Does not recommend illegal actions.
- Does not ask for or store raw bank credentials.
- Stores hashes/metadata where possible (e.g. watchlist patterns).
- If consent disallows, sensitive text in explanation payload is redacted.
- If model confidence is low, recommends a clarification question instead of escalating.
