# Anchor: Complete setup checklist

Use this checklist to get the **API**, **Supabase**, **web app**, and optional services working end-to-end. Do not commit real secrets; use `.env` (gitignored) or your deployment env.

---

## 1. Supabase: project and schema

### 1.1 Create project

1. [supabase.com/dashboard](https://supabase.com/dashboard) → **New project** (org, name, **save database password**).
2. Wait for the project to be ready.

### 1.2 Run bootstrap (once)

1. **SQL Editor** in the dashboard.
2. Open **`db/bootstrap_supabase.sql`** from this repo and run it **in full** (creates enums, tables 001–007, RLS, `user_household_id()`, etc.).
3. If you already ran an older bootstrap and only need newer objects, skip this and run the migrations below instead.

### 1.3 Run post-bootstrap migrations (in order)

The bootstrap file does **not** include migrations **008–014**. Run these **after** bootstrap, in **numeric order**, in SQL Editor (or via `scripts/run_migration.py` if `DATABASE_URL` is set):

| Order | File | Purpose |
|-------|------|--------|
| 1 | `008_pgvector_embeddings.sql` | pgvector extension + similarity search (optional but used for similar incidents when enabled) |
| 2 | `009_rings.sql` | `rings`, `ring_members` (Ring Discovery agent) |
| 3 | `010_household_calibration_params.sql` | `household_calibration` columns for calibration report |
| 4 | `011_role_consent_helpers.sql` | `user_can_contact()` for outreach RLS |
| 5 | `012_outbound_actions_caregiver_contacts.sql` | `outbound_actions`, `caregiver_contacts` (outreach) |
| 6 | `013_action_playbooks_capabilities_incident.sql` | `household_capabilities`, `action_playbooks`, `incident_packets` (if you use playbooks/incident) |
| 7 | `013_outbound_contact_safe_display.sql` | Safe display / RLS for outbound (run after 012) |
| 8 | `014_narrative_reports.sql` | `narrative_reports` (Evidence Narrative “View report”) |

**How to run:**

- **Option A – SQL Editor:** Open each file under `db/migrations/`, copy contents, run in SQL Editor.
- **Option B – Script:** From repo root, set `DATABASE_URL` (see below), then:
  ```bash
  python scripts/run_migration.py 008_pgvector_embeddings
  python scripts/run_migration.py 009_rings
  # ... etc for 010, 011, 012, 013_action_playbooks..., 013_outbound_contact..., 014_narrative_reports
  ```

**Note:** If a migration fails with “already exists”, that object is already there (e.g. from bootstrap); you can skip that statement or run the rest of the file.

### 1.4 Get Supabase credentials

In **Project Settings → API**:

- **Project URL** → use as `SUPABASE_URL`
- **anon public** key → use as `SUPABASE_ANON_KEY` (for frontend auth)
- **service_role** key (secret) → use as `SUPABASE_SERVICE_ROLE_KEY` (for API/worker; never expose in frontend)

In **Project Settings → Database** (optional, for running migrations from your machine):

- **Connection string** (URI) → use as `DATABASE_URL`. Use “Connection pooling” if available. Encode special characters in the password (e.g. `#` → `%23`).

---

## 2. Environment variables

### 2.1 Where to put them

- **API and worker:** Repo root **`.env`** or **`apps/api/.env`**. The API loads `.env` from the working directory (repo root when using `./scripts/run_api.sh`).
- **Web app:** **`apps/web/.env.local`** for Next.js (see apps/web/README.md).

### 2.2 Required for API and worker

| Variable | Description | Where to get it |
|----------|-------------|------------------|
| `SUPABASE_URL` | Supabase project URL | Project Settings → API → Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role secret | Project Settings → API → service_role |

Without these, the API returns **503 Supabase not configured** for routes that use Supabase.

### 2.3 Required for web app (non-demo)

| Variable | Description | Where to get it |
|----------|-------------|------------------|
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI base URL | e.g. `http://localhost:8000` (or your deployed API URL) |
| `NEXT_PUBLIC_SUPABASE_URL` | Same as Supabase project URL | Same as `SUPABASE_URL` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Anon/public key | Project Settings → API → anon public |

Without these, login/signup will show “Auth not configured” and API calls may fail.

### 2.4 Optional: migrations from your machine

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Postgres connection string (e.g. `postgresql://postgres.[ref]:[PASSWORD]@...pooler...:6543/postgres`) |

Set this to use `scripts/run_migration.py`; otherwise run each migration SQL manually in Supabase SQL Editor.

### 2.5 Optional: GNN / ML

| Variable | Description |
|----------|-------------|
| `ANCHOR_ML_CONFIG` | Path to ML config YAML (default `configs/hgt_baseline.yaml`) |
| `ANCHOR_ML_CHECKPOINT_PATH` | Path to trained checkpoint (e.g. `runs/.../best.pt`) |

When the checkpoint is missing, risk scoring uses rule-only fallback; similar incidents and embedding-centroid watchlists are unavailable (API returns `model_available: false` / `available: false`).

### 2.6 Optional: Neo4j (graph visualization)

| Variable | Description |
|----------|-------------|
| `NEO4J_URI` | Neo4j connection URI |
| `NEO4J_USER` | Neo4j user |
| `NEO4J_PASSWORD` | Neo4j password |

Used for “Sync to Neo4j” and Graph view; not required for core API or pipeline.

### 2.7 Optional: Evidence Narrative / LLM

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |

Used by Evidence Narrative agent for optional LLM narrative/hypotheses. If unset, the agent still runs with deterministic narrative only.

### 2.8 Optional: Outreach (real send)

| Variable | Description |
|----------|-------------|
| `ANCHOR_NOTIFY_PROVIDER` | `mock` (default), `twilio`, `sendgrid`, `smtp` |
| Twilio | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM` |
| SendGrid | `SENDGRID_API_KEY`, `SENDGRID_FROM` |
| SMTP | `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` |

Default is mock (no real send). Set provider and credentials to send real SMS/email.

### 2.9 Optional: Plaid (connectors)

| Variable | Description |
|----------|-------------|
| `PLAID_CLIENT_ID` | Plaid client ID |
| `PLAID_SECRET` | Plaid secret |

Only needed if you use the Plaid connector endpoints.

---

## 3. Auth and first user/household

### 3.1 Enable Auth provider

In Supabase: **Authentication → Providers** → enable **Email** (or others you need). For quick testing you can turn off “Confirm email”.

### 3.2 Link user to household

The API expects each authenticated user to have a row in **`public.users`** with `id = auth.uid()` and a **`household_id`**. RLS uses **`public.user_household_id()`** (reads from `users`).

**Recommended: use the app (no DB trigger)**

1. Sign up via the web app (**Create account**).
2. After sign-up, the app calls **`POST /households/onboard`** with the user’s JWT; the API creates a household and a `users` row (role `caregiver`).
3. If the user has no `users` row when they sign in, the app redirects to **Set up your household** (`/onboard`).

**Alternative: manual seed (SQL Editor)**

After creating a user in **Authentication → Users** (or via sign-up), copy their **UUID**, then run (replace `USER_UUID`):

```sql
INSERT INTO households (id, name) VALUES (gen_random_uuid(), 'My Household');

INSERT INTO users (id, household_id, role, display_name)
VALUES (
  'USER_UUID',
  (SELECT id FROM households ORDER BY created_at DESC LIMIT 1),
  'caregiver',
  'First User'
);
```

**If sign-up returns 500:** See [docs/SUPABASE_SETUP.md](SUPABASE_SETUP.md) (e.g. trigger on `auth.users`, email/SMTP, or missing `households`/`users` tables). Quick fix: run **`db/drop_signup_trigger.sql`** if you had a sign-up trigger, and use **`POST /households/onboard`** from the app instead.

---

## 4. Running the stack

### 4.1 API (FastAPI)

From **repo root**:

```bash
# With venv
python3 -m venv .venv
.venv/bin/pip install -e ".[ml]"   # or without [ml] for no PyTorch
./scripts/run_api.sh
```

Or:

```bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/apps/api"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- API base: **http://localhost:8000**
- Health: **GET http://localhost:8000/health**
- Docs: **http://localhost:8000/docs**

Ensure **`.env`** (or `apps/api/.env`) in the working directory has **`SUPABASE_URL`** and **`SUPABASE_SERVICE_ROLE_KEY`**.

### 4.2 Web app (Next.js)

```bash
cd apps/web
cp .env.example .env.local   # if you have an example
# Edit .env.local: NEXT_PUBLIC_API_BASE_URL, NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY
npm install
npm run dev
```

- App: **http://localhost:3000**
- Set **`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`** so the app talks to your API.

### 4.3 Worker (optional: pipeline + persist)

From **repo root**, with same `.env` as API:

```bash
./scripts/run_worker.sh --once --household-id <YOUR_HOUSEHOLD_UUID>
```

This runs ingest → graph build (using `domain.graph_service.build_graph_from_events`) → risk scoring → explain → watchlist → persist. For a real run you need events in `events` for that household (e.g. from ingest or seed).

### 4.4 Agent cron (optional)

```bash
ANCHOR_AGENT_CRON_DRY_RUN=false python scripts/run_agent_cron.py --household-id <UUID> --agent drift
# Or narrative, ring, calibration, redteam
```

Uses **`SUPABASE_URL`** and **`SUPABASE_SERVICE_ROLE_KEY`** (or anon).

---

## 5. Supabase tables and features (reference)

| Feature | Tables / artifacts | Migrations / bootstrap |
|--------|--------------------|-------------------------|
| Core (sessions, events, risk_signals, watchlists) | `households`, `users`, `sessions`, `events`, `entities`, `utterances`, `mentions`, `relationships`, `risk_signals`, `watchlists`, `summaries`, `feedback`, `agent_runs` | Bootstrap 001–007 |
| Similar incidents / embeddings | `risk_signal_embeddings` | Bootstrap 003, 004, 007; 008 for pgvector |
| Rings (Ring Discovery) | `rings`, `ring_members` | 009 |
| Calibration report | `household_calibration` | Bootstrap 003/004; 010 for params |
| Outreach | `outbound_actions`, `caregiver_contacts` | 011, 012 |
| Playbooks / incident packets | `household_capabilities`, `action_playbooks`, `incident_packets` | 013_action_playbooks... |
| Narrative report (“View report”) | `narrative_reports` | 014 |

RLS on all of these uses **`public.user_household_id()`** (and **`public.user_can_contact()`** for outbound_actions). Ensure **`users`** has the correct **`household_id`** for each auth user.

---

## 6. Verification

### 6.1 API + Supabase

1. **Health:** `curl http://localhost:8000/health` → `{"status":"ok"}`.
2. **Auth:** Sign in on the web app, then open a request that needs auth (e.g. **GET /agents/status**). In browser dev tools or a REST client, use the Supabase session access token as **`Authorization: Bearer <token>`**. You should get 200 and JSON (not 401/503).
3. **Household:** After onboarding, **Table Editor → users** and **households** should have one row each for your user/household.

### 6.2 Web app

1. Open **http://localhost:3000**, sign up or sign in, complete onboarding.
2. **Dashboard:** “Today” and quick links (Alerts, Rings, Reports, Replay, Agents) load.
3. **Alerts:** List loads (or empty). Open an alert if any; similar incidents and graph should load or show “Unavailable” when no embeddings.
4. **Rings:** **Rings** in nav → list (or “No rings yet”); run Ring Discovery agent from Agents, then refresh.
5. **Reports:** **Reports** in nav → index; open **Calibration** or **Red-team** (may 404 until you run the corresponding agent).
6. **Agents:** **Agents** → status for each agent; run an agent (e.g. Narrative, Drift, Ring) with dry run, then “View report” / “Copy retrain command” / Rings as applicable.

### 6.3 Optional: worker and GNN

1. Ingest some events (e.g. **POST /ingest/events** or seed script), then run worker once with your `household_id`. Check **risk_signals** and **agent_runs** in Table Editor.
2. If you have a trained checkpoint, set **`ANCHOR_ML_CHECKPOINT_PATH`** and run the pipeline again; similar incidents and embedding watchlists should become available when the model runs.

---

## 7. Common issues

| Symptom | What to check |
|--------|----------------|
| API 503 “Supabase not configured” | `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env` (repo root or `apps/api`), and API started from that directory. |
| 401 on API with token | Token is valid Supabase JWT; user exists in **auth.users**; **public.users** has a row with that `id` and a valid `household_id`. |
| 404 “Not onboarded” | User has no **public.users** row or no `household_id`. Use **POST /households/onboard** (with JWT) or manual seed. |
| Sign-up 500 | Supabase Auth issue: trigger on `auth.users`, SMTP, or schema. See [SUPABASE_SETUP.md](SUPABASE_SETUP.md). |
| “relation … does not exist” | Run bootstrap and migrations 008–014 so all tables exist. |
| RLS policy violation | `user_household_id()` returns the user’s household; that user must have a **users** row with correct **household_id**. |
| Similar incidents “Unavailable” | No GNN checkpoint or no embeddings persisted for that signal; expected when model did not run. |
| Rings / Reports empty | Run the Ring Discovery or Calibration/Red-team/Narrative agent at least once (with dry_run=false to persist). |

---

## 8. Summary checklist

- [ ] Supabase project created; **bootstrap_supabase.sql** run once.
- [ ] Migrations **008–014** run in order (pgvector, rings, calibration params, role/consent, outbound, playbooks if needed, narrative_reports).
- [ ] **.env** (repo root or apps/api): `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`; optionally `DATABASE_URL`, `ANCHOR_ML_CHECKPOINT_PATH`, Neo4j, OpenAI, notify, Plaid.
- [ ] **apps/web/.env.local**: `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
- [ ] Auth provider (e.g. Email) enabled; first user linked to household via app onboard or manual seed.
- [ ] API starts without 503; **GET /health** and **GET /agents/status** with JWT succeed.
- [ ] Web app loads; sign-in and onboarding work; Dashboard, Alerts, Rings, Reports, Agents and Replay are reachable and behave as expected.

For more detail: [SUPABASE_SETUP.md](SUPABASE_SETUP.md), [README.md](../README.md), [QUICKSTART_API.md](QUICKSTART_API.md).
