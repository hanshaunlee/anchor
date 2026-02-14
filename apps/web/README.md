# Anchor Web

Next.js 14 (App Router) frontend for **Anchor** — offline-first elder companion and graph risk engine. Integrates with the FastAPI + Supabase backend.

## Stack

- **Next.js 14** (App Router) + TypeScript
- **TailwindCSS** + **shadcn/ui**
- **Zustand** (client state)
- **TanStack Query** (API + caching)
- **React Flow** (@xyflow/react) for graph visualization
- **Recharts** for charts
- **Framer Motion** for animations
- **Supabase Auth** for login/session (all data reads go through FastAPI)
- **WebSocket** client to `/ws/risk_signals` for live risk signals (fallback: polling)

## Setup

### Environment variables

Create `.env.local` in `apps/web`:

```bash
# Backend API (required for non-demo)
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Supabase (required for auth)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Optional: demo mode (use fixtures, no API)
NEXT_PUBLIC_DEMO_MODE=false

# Optional: app URL for server-side fixture fetch in demo
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

- **API_BASE_URL**: Base URL of the Anchor FastAPI app (e.g. `http://localhost:8000`).
- **SUPABASE_URL** / **SUPABASE_ANON_KEY**: For Supabase Auth. If not set, login will show “Auth not configured”; you can still use Dashboard in demo mode.
- **DEMO_MODE**: When `true`, the app loads data from `/fixtures/*.json` and runs fully offline for demos.
- **APP_URL**: Used when fetching fixtures from the server (e.g. SSR); default `http://localhost:3000`.

### Install and run

```bash
cd apps/web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

**Run API + frontend together:** See [docs/QUICKSTART_API.md](../../docs/QUICKSTART_API.md) — API in one terminal, `npm run dev` in another (from repo root or `apps/web`).

### Build

```bash
npm run build
npm run start
```

## Routes

| Route | Description |
|-------|-------------|
| `/` | Landing; links to Sign in, Dashboard, Scenario Replay |
| `/login` | Supabase sign in |
| `/signup` | Create account (Supabase Auth); then use `/onboard` or API `POST /households/onboard` to create household |
| `/onboard` | Set up household after sign-up (display name, household name); calls `POST /households/onboard` |
| `/logout` | Sign out and redirect to `/` |
| `/dashboard` | Caregiver home: today feed, risk chart, latest signals |
| `/alerts` | Filterable risk signals list (real-time via WS) |
| `/alerts/[id]` | Investigation: timeline, graph evidence, similar incidents, feedback, agent trace |
| `/sessions` | Sessions list (date range) |
| `/sessions/[id]` | Session detail: summary, events table, consent banner |
| `/watchlists` | Watchlists with priority, reason, expiry |
| `/summaries` | Weekly summaries and trend charts |
| `/ingest` | Event ingest (device batch upload) |
| `/graph` | Graph view: evidence subgraph, Sync to Neo4j |
| `/agents` | Agent Center: pipeline steps, dry run, trace |
| `/elder` | Elder view: big text, today summary, share toggle |
| `/replay` | Scenario Replay: animate scam storyline (score chart, graph, trace) |

## Demo mode

Set `NEXT_PUBLIC_DEMO_MODE=true` or toggle “Demo mode” in the dashboard sidebar. Data is then loaded from `public/fixtures/`:

- `household_me.json`
- `risk_signals.json`
- `risk_signal_detail.json`
- `sessions.json`
- `session_events.json`
- `watchlists.json`
- `summaries.json`
- `scenario_replay.json` (used by `/replay`)

Scenario Replay uses `scenario_replay.json` to animate risk score over time, graph node highlights, and agent trace steps.

## API contract

The app uses only the endpoints and shapes defined in the repo:

- **[docs/api_ui_contracts.md](../../docs/api_ui_contracts.md)** — REST, WebSocket, and JSON shapes (households/me, onboard, sessions, risk_signals, similar, feedback, watchlists, device/sync, ingest, summaries, agents).
- **`apps/api/api/routers/*`** — route handlers.
- **`apps/api/api/schemas.py`** — Pydantic request/response models.

See **[docs/frontend_notes.md](../../docs/frontend_notes.md)** for a summary of data objects (HouseholdMe, Session, Event, RiskSignal, Watchlist, Summary, Feedback, WebSocket message).

## Role-based UI

- **Elder**: `/elder` shows minimal summary, recommendation, “Share with caregiver” toggle, and recent interactions (no raw text when redacted).
- **Caregiver / Admin**: Full dashboard, alerts, graph evidence, feedback, agent trace, sessions, watchlists, summaries.
