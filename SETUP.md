# Anchor: Setup

Single guide to get the **API**, **Supabase**, **web app**, and optional services working. Do not commit real secrets; use `.env` (gitignored) or your deployment env.

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

Bootstrap does **not** include migrations **008–024**. Run these **after** bootstrap, in **numeric order**, in SQL Editor (or via `scripts/run_migration.py` if `DATABASE_URL` is set):

| Order | File | Purpose |
|-------|------|--------|
| 1 | `008_pgvector_embeddings.sql` | pgvector + similarity search (similar incidents when enabled) |
| 2 | `009_rings.sql` | `rings`, `ring_members` (Ring Discovery) |
| 3 | `010_household_calibration_params.sql` | `household_calibration` for calibration report |
| 4 | `011_role_consent_helpers.sql` | `user_can_contact()` for outreach RLS |
| 5 | `012_outbound_actions_caregiver_contacts.sql` | `outbound_actions`, `caregiver_contacts` |
| 6 | `013_action_playbooks_capabilities_incident.sql` | `household_capabilities`, `action_playbooks`, `incident_packets` |
| 7 | `013_outbound_contact_safe_display.sql` | Safe display / RLS for outbound (run after 012) |
| 8 | `014_narrative_reports.sql` | `narrative_reports` (Evidence Narrative "View report") |
| 9 | `015_outbound_actions_conformal_auto_send.sql` | Conformal auto-send and outreach columns |
| 10 | `016_rpc_alert_page_and_investigation_context.sql` | RPCs for alert page and investigation |
| 11 | `017_performance_indexes.sql` | Performance indexes |
| 12 | `018_processing_queue.sql` | `processing_queue` (enqueued investigation) |
| 13 | `019_processing_queue_dedupe_retry.sql` | Dedupe and retry for processing_queue |
| 14 | `020_watchlist_items.sql` | `watchlist_items` |
| 15 | `021_rings_fingerprint_canonical.sql` | Rings fingerprint and canonical view |
| 16 | `022_embedding_vector_128.sql` | 128-dim embedding vector support |
| 17 | `023_protection_rings_watchlist_columns.sql` | Protection page rings/watchlist columns |
| 18 | `024_risk_signals_fingerprint.sql` | `risk_signals.fingerprint` for compound upsert |

**How to run:** Open each file under `db/migrations/`, copy contents, run in SQL Editor. Or from repo root with `DATABASE_URL` set: `python scripts/run_migration.py 008_pgvector_embeddings`, then 009 … through 024 in order.

If a migration fails with "already exists", skip that statement or run the rest of the file.

**Shortcut for alerts/outreach only:** You can run **`db/run_migrations_011_012_013.sql`** in SQL Editor to create the tables needed for Notify caregiver and Action plan (011–013); then run the remaining migrations 014–024 in order when you want full features.

### 1.4 Get Supabase credentials

In **Project Settings → API**:

- **Project URL** → `SUPABASE_URL`
- **anon public** key → `SUPABASE_ANON_KEY` / `NEXT_PUBLIC_SUPABASE_ANON_KEY` (frontend auth)
- **service_role** key (secret) → `SUPABASE_SERVICE_ROLE_KEY` (API/worker only; never in frontend)

In **Project Settings → Database** (optional, for running migrations from your machine):

- **Connection string** (URI) → `DATABASE_URL`. Use "Connection pooling" if available. Encode special characters in the password (e.g. `#` → `%23`).

---

## 2. Environment variables

### 2.1 Where to put them

- **API and worker:** Repo root **`.env`** or **`apps/api/.env`**. The API loads `.env` from the working directory (repo root when using `./scripts/run_api.sh`).
- **Web app:** **`apps/web/.env.local`** for Next.js.

### 2.2 Required for API and worker

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL (Project Settings → API → Project URL) |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role secret (Project Settings → API → service_role) |

Without these, the API returns **503 Supabase not configured**.

### 2.3 Required for web app (non-demo)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI base URL, e.g. `http://localhost:8000` |
| `NEXT_PUBLIC_SUPABASE_URL` | Same as Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Anon/public key (Project Settings → API → anon public) |

Without these, login/signup shows "Auth not configured" and API calls may fail.

### 2.4 Optional

| Variable(s) | Purpose |
|-------------|---------|
| `DATABASE_URL` | Postgres connection string for `scripts/run_migration.py` |
| `ANCHOR_ML_CONFIG`, `ANCHOR_ML_CHECKPOINT_PATH` | GNN/ML; when missing, rule-only fallback; similar incidents and embedding watchlists unavailable |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph view "Sync to Neo4j" and Neo4j Browser (see section 9) |
| `OPENAI_API_KEY` | Evidence Narrative agent optional LLM narrative |
| `ANTHROPIC_API_KEY` | Explain API (`POST /explain`) plain-language IDs for caregivers; [console.anthropic.com](https://console.anthropic.com) |
| `ANCHOR_NOTIFY_PROVIDER` + Twilio/SendGrid/SMTP vars | Real SMS/email (default `mock`) |
| `PLAID_CLIENT_ID`, `PLAID_SECRET` | Plaid connector endpoints only |

---

## 3. Auth and first user/household

### 3.1 Enable Auth provider

In Supabase: **Authentication → Providers** → enable **Email** (or others). For quick testing you can turn off "Confirm email".

### 3.2 Link user to household

The API expects each authenticated user to have a row in **`public.users`** with `id = auth.uid()` and a **`household_id`**. RLS uses **`public.user_household_id()`**.

**Recommended: use the app (no DB trigger)**

1. Sign up via the web app (**Create account**).
2. After sign-up, the app calls **`POST /households/onboard`**; the API creates a household and a `users` row (role `caregiver`).
3. If the user has no `users` row when they sign in, the app redirects to **Set up your household** (`/onboard`).

**Alternative: manual seed (SQL Editor)**

After creating a user in **Authentication → Users**, copy their **UUID**, then run (replace `USER_UUID`):

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

### 3.3 Sign-up returns 500

If **Create account** returns 500 and Supabase shows `unexpected_failure` on `/auth/v1/signup`, the failure is in **Supabase Auth**. Common causes:

1. **Trigger on `auth.users`** — Run **`db/drop_signup_trigger.sql`** in SQL Editor so signup no longer runs it; the app will create household/user via `POST /households/onboard`. Ensure **`db/repair_households_users.sql`** has been run if those tables were missing.
2. **Email / SMTP** — If "Confirm email" is on and sending fails, Auth can return 500. For quick testing, turn off **Authentication → Providers → Email → Confirm email**.
3. **`relation "households" does not exist`** — Run **`db/repair_households_users.sql`** (creates enum, `households`, `users` if missing), or run full **`db/bootstrap_supabase.sql`** if you never bootstrapped.

---

## 4. Running the stack

### 4.1 API + web (run both)

**Terminal 1 – API** (from repo root):

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ml]"   # or without [ml] for no PyTorch
./scripts/run_api.sh
```

Or: `export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/apps/api"` then `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`.

- API: **http://localhost:8000** — Health: **GET /health**, Docs: **http://localhost:8000/docs**

**Terminal 2 – Web** (from repo root or `apps/web`):

```bash
cd apps/web
cp .env.example .env.local   # if present
# Edit .env.local: NEXT_PUBLIC_API_BASE_URL, NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY
npm install
npm run dev
```

- App: **http://localhost:3000** — Set **`NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`** so the app talks to your API.

**Demo mode (no API):** Toggle "Demo mode" in the app or set `NEXT_PUBLIC_DEMO_MODE=true`; the app uses local fixtures.

### 4.2 Worker (optional)

From **repo root**, with same `.env` as API:

```bash
./scripts/run_worker.sh --once --household-id <YOUR_HOUSEHOLD_UUID>
```

Runs ingest → graph build → risk scoring → explain → watchlist → persist. You need events in `events` for that household (e.g. from ingest or seed).

### 4.3 Agent cron (optional)

```bash
ANCHOR_AGENT_CRON_DRY_RUN=false python scripts/run_agent_cron.py --household-id <UUID> --agent drift
# Or narrative, ring, calibration, redteam
```

---

## 5. Notify caregiver & Action plan (optional)

To use **Notify caregiver** and **Run Incident Response** (action plan) from the Alerts page:

1. **Tables:** Run migrations 011–013 (or full 008–024). Quick option: run **`db/run_migrations_011_012_013.sql`** in SQL Editor.
2. **One-time setup:** Run the setup script so consent and a caregiver contact exist:

   ```bash
   SEED_USER_ID=<auth-user-uuid> PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py
   ```

   Or with **HOUSEHOLD_ID** if you don't have the user UUID:  
   `HOUSEHOLD_ID=<household-uuid> PYTHONPATH=apps/api:. python scripts/enable_alerts_setup.py`  
   Optional: `CAREGIVER_EMAIL=you@example.com` to use your email for the demo contact.

3. **Use the app:** Sign in as a caregiver, open **Alerts** → one alert. Use **Notify caregiver** (Preview → Confirm send) and **Run Incident Response**; tasks appear and you can Mark done.

**Manual alternative:** Set household/session consent (`allow_outbound_contact` / `consent_allow_outbound_contact`) and add a row to `caregiver_contacts` for the household (via API `PATCH /households/me/consent`, `POST /households/me/contacts` or via SQL). See **Common issues** below if Notify or Action plan fail.

---

## 6. Verification

- **API + Supabase:** `curl http://localhost:8000/health` → `{"status":"ok"}`. Call **GET /agents/status** with a valid Bearer JWT (Supabase session token). Table Editor → `users` and `households` should have your user/household after onboarding.
- **Web:** Open http://localhost:3000, sign in, complete onboarding. Dashboard, Alerts, Rings, Reports, Agents, Replay should load (or empty state).
- **Optional:** Ingest events, run worker once; with a trained checkpoint set `ANCHOR_ML_CHECKPOINT_PATH` and run again to see similar incidents and embedding watchlists.

---

## 7. Common issues

| Symptom | What to check |
|--------|----------------|
| API 503 "Supabase not configured" | `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env` (repo root or `apps/api`); API started from that directory. |
| 401 on API with token | Valid Supabase JWT; user in **auth.users**; **public.users** has a row with that `id` and valid `household_id`. |
| 404 "Not onboarded" | No **public.users** row or no `household_id`. Use **POST /households/onboard** (with JWT) or manual seed. |
| Sign-up 500 | See **3.3 Sign-up returns 500** (trigger, SMTP, or missing tables). |
| "relation … does not exist" | Run bootstrap and migrations 008–024 so all tables exist. |
| RLS policy violation | `user_household_id()` must return the user's household; **users** row must have correct **household_id**. |
| Similar incidents "Unavailable" | No GNN checkpoint or no embeddings for that signal; expected when model did not run. |
| Notify: "Consent does not allow outbound contact" | Run `enable_alerts_setup.py` or set consent (API or SQL) as in section 5. |
| Notify: no recipient / always demo@example.com | Add a caregiver contact (script or `POST /households/me/contacts`). |
| Action plan / Run Incident Response does nothing | Ensure migrations 013 (action_playbooks, action_tasks, incident_packets) are applied; check network tab for 4xx/5xx. |

---

## 8. Summary checklist

- [ ] Supabase project created; **`db/bootstrap_supabase.sql`** run once.
- [ ] Migrations **008–024** run in order (or at least 011–013 for alerts; then 014–024 when needed).
- [ ] **.env** (root or `apps/api`): `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`; optional vars as in section 2.
- [ ] **apps/web/.env.local**: `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
- [ ] Auth provider (e.g. Email) enabled; first user linked to household via app onboard or manual seed.
- [ ] API starts without 503; **GET /health** and **GET /agents/status** with JWT succeed.
- [ ] Web app loads; sign-in and onboarding work; Dashboard, Alerts, Rings, Reports, Agents, Replay behave as expected.
- [ ] (Optional) Notify caregiver + Action plan: run `enable_alerts_setup.py` once with **SEED_USER_ID** or **HOUSEHOLD_ID**.

---

## 9. Optional: Neo4j (graph visualization)

Neo4j is **optional**. When enabled, **Graph view** in the dashboard can sync the evidence graph to Neo4j for Cypher and Neo4j Browser.

**Quick start (Docker):** Start Docker, then from repo root:

```bash
./scripts/start_neo4j.sh
```

Default password is `neo4j123`. Add to `.env` (repo root or `apps/api/.env`):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j123
```

Restart the API. In the app: **Graph view** → **Sync to Neo4j** → **Open in Neo4j Browser**. (Neo4j 5 does not allow password `neo4j`; use `neo4j123` or run Neo4j 4 with `NEO4J_IMAGE=neo4j:4 NEO4J_PASSWORD=neo4j ./scripts/start_neo4j.sh`.)

**Without Docker:** Install [Neo4j Desktop](https://neo4j.com/download/) or `brew install neo4j` then `neo4j console`. Same `.env`; Browser at http://localhost:7474/browser.

**API:** Install driver: `pip install neo4j`. `GET /graph/neo4j-status` returns `{"enabled": true, "browser_url": "..."}` when configured. If `NEO4J_URI` is unset, Neo4j is disabled and sync/browser buttons are hidden.

---

For repo layout and reference see [README.md](README.md) and [README_EXTENDED.md](README_EXTENDED.md).
