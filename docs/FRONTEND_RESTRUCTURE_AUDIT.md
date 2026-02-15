# Frontend Restructure Audit

Audit for restructuring the Anchor frontend: components to keep, data/function flow, demo mode, and agents (coded vs implemented). Use this to preserve all functionality and tie the codebase together.

---

## 1. Components to Keep (by category)

### 1.1 Layout & navigation
| Component / File | Purpose | Notes |
|-----------------|--------|------|
| `app/(dashboard)/layout.tsx` | Dashboard shell, sidebar, demo toggle, sign out | **Demo mode**: toggle in footer; when ON, skips onboard redirect. Keep. |
| `components/dashboard-nav.tsx` | Nav links (Today, Alerts, Graph, Sessions, Watchlists, Rings, Summaries, Reports, Automation Center, Ingest, Replay, Settings, Elder view) | Single source for nav; consider grouping by "Overview / Alerts / Evidence / Automation / Data / Settings". |
| `app/layout.tsx` | Root layout | Keep. |
| `app/page.tsx` | Landing / redirect | Keep. |
| `app/(auth)/login/page.tsx`, `signup/page.tsx`, `signup/success/page.tsx`, `onboard/page.tsx`, `logout/page.tsx` | Auth flows | Keep. |

### 1.2 Dashboard & alerts
| Component / File | Purpose | Notes |
|-----------------|--------|------|
| `app/(dashboard)/dashboard/page.tsx` | Today: open alerts count, severity distribution, 7-day risk chart, latest risk signals, quick links | Uses `useRiskSignals`, `useHouseholdMe`, `useRiskSignalStream`. Keep. |
| `app/(dashboard)/alerts/page.tsx` | Alerts list | Keep. |
| `app/(dashboard)/alerts/[id]/page.tsx` | Alert detail wrapper | Keep. |
| `app/(dashboard)/alerts/[id]/alert-detail-content.tsx` | **Full alert detail**: summary, motifs, evidence, graph, similar incidents, **Refresh Investigation**, **Notify caregiver** (preview/send), **Action Plan** (playbook + Incident Response), feedback, deep-dive explain, policy gate | Heavy; many hooks. **Demo**: hides Refresh, Notify, Action Plan; shows demo banner. Keep all sections. |
| `components/risk-signal-card.tsx` | Card for one risk signal in lists | Keep. |

### 1.3 Automation Center (agents page)
| Component / File | Purpose | Notes |
|-----------------|--------|------|
| `app/(dashboard)/agents/page.tsx` | **Automation Center**: tabs Investigation / Actions / System Health / Advanced (admin). Investigation run (dry run, use demo events), run result, trace, artifact links; Actions: outreach candidates, preview/send, manual alert ID, history; System: Model Health run; Dev: individual agents (drift, narrative, ring, calibration, redteam) | **Demo**: run investigation uses mock result; no API. **Gap**: no visible "agents behind the scenes" (e.g. Financial → Narrative → Outreach); no standalone Financial run or demo from here. Keep all tabs and add agent-story/demo. |
| `components/run-result-summary.tsx` | Summary of investigation run (created/updated signals, watchlists, outreach count, step trace, links) | Keep. |
| `components/trace-viewer.tsx` | Wraps step_trace array into `AgentTrace` | Keep. |
| `components/agent-trace.tsx` | Renders pipeline steps (success/warn/fail) | Keep. |
| `components/artifact-links.tsx` | Links to created alerts, watchlists, outreach candidates | Keep. |
| `components/actions-history.tsx` | Outreach actions list with status filter | Keep. |
| `components/consent-gate-banner.tsx` | Consent/contact missing for outreach | Keep. |
| `components/model-health-card.tsx` | Model Health last run, Run maintenance button | Keep. |

### 1.4 Evidence & graph
| Component / File | Purpose | Notes |
|-----------------|--------|------|
| `app/(dashboard)/graph/page.tsx` | Graph view (evidence subgraph, Neo4j sync) | Keep. |
| `components/graph-evidence.tsx` | Renders evidence subgraph (nodes/edges) | Keep. |
| `app/(dashboard)/sessions/page.tsx`, `sessions/[id]/page.tsx`, `session-detail-content.tsx` | Sessions list and detail + events | Keep. |
| `app/(dashboard)/watchlists/page.tsx` | Watchlists | Keep. |
| `app/(dashboard)/rings/page.tsx`, `rings/[id]/page.tsx` | Ring Discovery artifacts | Keep. |
| `app/(dashboard)/summaries/page.tsx` | Summaries | Keep. |

### 1.5 Reports
| Component / File | Purpose | Notes |
|-----------------|--------|------|
| `app/(dashboard)/reports/page.tsx` | Reports hub (narrative, calibration, redteam) | Keep. |
| `app/(dashboard)/reports/narrative/[id]/page.tsx` | Evidence Narrative report by id | Uses `api.getNarrativeReport(id)`. Keep. |
| `app/(dashboard)/reports/calibration/page.tsx` | Calibration report (ECE, precision/recall) | Uses `api.getCalibrationReport()`. Keep. |
| `app/(dashboard)/reports/redteam/page.tsx` | Red-team report (pass rate, failing cases, Open in replay) | Uses `api.getRedteamReport()`. Keep. |

### 1.6 Replay, ingest, settings, elder
| Component / File | Purpose | Notes |
|-----------------|--------|------|
| `app/(dashboard)/replay/page.tsx` | Scenario replay: fixture or API (`getFinancialDemo`), timeline + graph + agent trace | **Demo**: no "Load from API"; uses fixture. Redteam source links to replay. Keep. |
| `app/(dashboard)/ingest/page.tsx` | Paste JSON events, submit ingest | **Demo**: page shows "Ingest disabled in demo mode". Keep. |
| `app/(dashboard)/settings/page.tsx` | Capabilities (SMS, email, device push, bank lock_card, enable_alerts), consent (allow_outbound_contact) | Keep. No demo toggle here (toggle is in layout). |
| `app/(dashboard)/elder/page.tsx` | Elder view | Keep. |

### 1.7 Policy & UI primitives
| Component / File | Purpose | Notes |
|-----------------|--------|------|
| `components/policy-gate-card.tsx` | Policy gate messaging | Keep. |
| `components/ui/*` | card, button, input, label, switch, tabs, badge, skeleton, scroll-area, separator, textarea, select | Keep. |
| `app/(dashboard)/error.tsx`, `app/error.tsx` | Error boundaries | Keep. |
| `providers/auth-provider.tsx`, `providers/query-provider.tsx` | Auth and React Query | Keep. |

---

## 2. Data & function flow

### 2.1 API client (`lib/api/client.ts`)
- **Household**: `getHouseholdMe`, `postOnboard`, `getHouseholdConsent`, `patchHouseholdConsent`
- **Sessions**: `getSessions`, `getSessionEvents`
- **Risk signals**: `getRiskSignals`, `getRiskSignal`, `getRiskSignalPage` (compound), `getSimilarIncidents`, `postDeepDiveExplain`, `submitFeedback`
- **Watchlists**: `getWatchlists`
- **Rings**: `getRings`, `getRing`
- **Summaries**: `getSummaries`
- **Agents**: `getAgentsStatus`, `getAgentsCatalog`, `postInvestigationRun`, `postAlertRefresh`, `postMaintenanceRun`, `postFinancialRun`, `getFinancialDemo`, `getFinancialTrace`, `getCalibrationReport`, `getRedteamReport`, `getNarrativeReport(reportId)`, `getAgentTrace(runId, agentName)`, `postAgentRun(slug)` (drift|narrative|ring|calibration|redteam), `postOutreachPreview`, `postOutreachSend`, `getOutreachActions`, `getOutreachAction`, `getOutreachCandidates`, `getOutreachSummary`, `postOutreachRun`, `postOutreach`
- **Playbooks / incident**: `getRiskSignalPlaybook`, `getPlaybook`, `completePlaybookTask`, `getIncidentPacket`, `postIncidentResponseRun`
- **Graph**: `getGraphEvidence`, `postGraphSyncNeo4j`, `getGraphNeo4jStatus`
- **Ingest**: `postIngestEvents`
- **Capabilities**: `getCapabilitiesMe`, `patchCapabilities`

### 2.2 Hooks (`hooks/use-api.ts`)
- **Demo branching**: `useHouseholdMe`, `useSessions`, `useSessionEvents`, `useRiskSignals`, `useRiskSignalDetail`, `useAlertPage`, `useSimilarIncidents`, `useWatchlists`, `useRings`, `useRing`, `useSummaries`, `useAgentsStatus`, `useAgentCatalog`, `useFinancialTrace`, `useAgentTrace`, `useGraphEvidence`, `useCapabilitiesMe`, `useHouseholdConsent` all branch on `demoMode` (fixtures vs API). Mutations `usePatchCapabilitiesMutation`, `usePatchHouseholdConsentMutation` in demo mode write to cache only.
- **Query keys**: Include `demoMode` where fixtures are used so toggling demo invalidates correctly.
- **Mutations and invalidation**: e.g. `useInvestigationRunMutation` invalidates agents status, risk_signals, watchlists, outreach; `useAlertRefreshMutation` invalidates alert page and related; `useOutreachSendMutation` invalidates outreach, risk_signals, agents status. Keep this map when restructuring.

### 2.3 Key user flows
1. **Investigation (supervisor)**  
   Automation Center → Run Investigation (optional: use demo events) → POST `/investigation/run` → Run result + trace + artifact links → user can open alerts, go to Actions tab for outreach.
2. **Single-alert refresh**  
   Alert detail → "Refresh Investigation" → POST `/alerts/:id/refresh` → invalidation → updated alert page.
3. **Outreach**  
   Automation Center Actions tab **or** Alert detail → Preview → (optional) Send → POST `/actions/outreach/preview`, `/actions/outreach/send` (or legacy `postOutreach` with dry_run then false). Consent and caregiver contact gated.
4. **Incident Response**  
   Alert detail → "Run Incident Response" → POST `/agents/incident-response/run` → playbook + tasks (e.g. call_bank script, complete task).
5. **Model Health**  
   Automation Center → System Health → Run maintenance → POST `/system/maintenance/run` (model_health agent).
6. **Dev agents**  
   Automation Center → Advanced → Run (dry / real) for drift, narrative, ring, calibration, redteam → POST `/agents/{slug}/run`.
7. **Replay**  
   Replay page → Load from API (calls `getFinancialDemo`) or use fixture → timeline + graph + agent trace.
8. **Reports**  
   Reports hub → Narrative (by id), Calibration (latest), Redteam (latest) → `getNarrativeReport`, `getCalibrationReport`, `getRedteamReport`.

---

## 3. Demo mode (nuances)

| Area | Behavior |
|------|----------|
| **Store** | `use-app-store.ts`: `demoMode` (default `NEXT_PUBLIC_DEMO_MODE === "true"`), `setDemoMode`. |
| **Layout** | Toggle "Demo mode ON/OFF" in sidebar; when demo, skip onboard redirect. |
| **Data** | All fixture-backed hooks use `/fixtures/*.json` when `demoMode`; query keys include `demoMode`. |
| **Automation Center** | Run Investigation: if demo, set mock `lastInvestigationResult` (no API). |
| **Alert detail** | Hide "Refresh Investigation", "Notify caregiver", "Action Plan"; show banner that demo is on. |
| **Ingest** | Show "Ingest disabled in demo mode". |
| **Replay** | Hide "Load from API"; only fixture. |
| **Traces** | `useFinancialTrace`, `useAgentTrace` disabled when demo (`enabled: !!runId && !demoMode`). |
| **Similar incidents** | In demo, return `{ available: true, similar: [] }` (no API). |

**Fixtures to keep** (in `public/fixtures/`):  
`household_me.json`, `sessions.json`, `session_events.json`, `risk_signals.json`, `risk_signal_detail.json`, `summaries.json`, `watchlists.json`, `agents_status.json`, `agents_catalog.json`, `capabilities_me.json`, `consent_me.json`, `graph_evidence.json`, `scenario_replay.json`.

---

## 4. Agents: registry vs implemented in UI

### 4.1 Backend registry (`domain/agents/registry.py`)

| Slug | Agent | Tier | Triggers | Visibility | Primary artifacts |
|------|--------|------|----------|------------|-------------------|
| financial | Financial Security | INVESTIGATION | AUTOMATIC, MANUAL_UI | DEFAULT_UI | risk_signals, watchlists |
| narrative | Evidence Narrative | INVESTIGATION | AUTOMATIC | HIDDEN | risk_signals, summaries |
| ring | Ring Discovery | INVESTIGATION | AUTOMATIC, ADMIN_ONLY | ADVANCED_UI | rings, risk_signals, watchlists |
| outreach | Caregiver Outreach | USER_ACTION | AUTOMATIC, MANUAL_UI | DEFAULT_UI | outbound_actions, risk_signals |
| supervisor | Investigation (Supervisor) | INVESTIGATION | AUTOMATIC, MANUAL_UI | DEFAULT_UI | risk_signals, watchlists, outreach_candidates |
| model_health | Model Health | SYSTEM_MAINTENANCE | SCHEDULED, ADMIN_ONLY | ADVANCED_UI | summaries, household_calibration, risk_signals |
| drift | Graph Drift | SYSTEM_MAINTENANCE | ADMIN_ONLY | ADVANCED_UI | risk_signals, summaries |
| calibration | Continual Calibration | SYSTEM_MAINTENANCE | ADMIN_ONLY | ADVANCED_UI | summaries, risk_signals, household_calibration |
| redteam | Synthetic Red-Team | DEVTOOLS | ADMIN_ONLY | ADVANCED_UI | risk_signals, replay_fixture, summaries |
| incident_response | Incident Response | USER_ACTION | MANUAL_UI, ADMIN_ONLY | ADVANCED_UI | action_playbooks, action_tasks, incident_packets |

### 4.2 Where each agent is used in the frontend

| Agent | Where used | Implemented? |
|-------|------------|---------------|
| **supervisor** | Automation Center: "Run Investigation"; Alert detail: "Refresh Investigation". Trace via `useAgentTrace(supervisorRunId, "supervisor")`. | Yes |
| **financial** | Backend: part of supervisor pipeline. **Standalone**: only via API `postFinancialRun` (not in UI). **Demo**: Replay page uses `getFinancialDemo` for "Load from API". | Partially (no run from Automation Center; only in Replay) |
| **narrative** | Runs inside supervisor (AUTOMATIC). Reports: Evidence Narrative report by id. Automation Center Dev: "Run Narrative" (dry/real). | Yes (report + dev run) |
| **ring** | Supervisor can invoke. Rings pages list/detail. Automation Center Dev: "Run Ring". | Yes |
| **outreach** | Automation Center Actions: candidates, preview/send; Alert detail: Notify caregiver. `postOutreachPreview` / `postOutreachSend`. | Yes |
| **model_health** | Automation Center System Health: "Run maintenance" → POST `/system/maintenance/run`. Status from `getAgentsStatus` (model_health). | Yes |
| **drift** | Part of model_health. Automation Center Dev: "Run Drift". | Yes (dev only) |
| **calibration** | Part of model_health. Automation Center Dev: "Run Calibration". Reports: calibration report page. | Yes |
| **redteam** | Automation Center Dev: "Run Red-Team". Reports: redteam report; "Open in replay" link. | Yes |
| **incident_response** | Alert detail: "Run Incident Response" → playbook + tasks. No Automation Center entry. | Yes (alert-only) |

### 4.3 Gaps (coded, not or barely surfaced)

1. **Financial Security as a visible step**  
   - No "Run Financial only" or "See Financial demo" in Automation Center.  
   - Replay page has "Load from API" = `getFinancialDemo()`; that’s the only live Financial demo in UI.  
   - **Suggestion**: In Automation Center, add a "How it works" or "Pipeline demo" that either runs `getFinancialDemo()` or shows a fixed story: Ingest → Financial (risk + watchlist) → Narrative → Outreach candidates, with links to Replay and to Alerts.

2. **Narrative visibility**  
   - Narrative runs inside Investigation; report is under Reports. No clear "Evidence Narrative ran for this investigation" in Automation Center.  
   - **Suggestion**: In run result or trace, show "Narrative ran for signals X, Y" and link to narrative report if one was created.

3. **Incident Response in Automation Center**  
   - Only triggered from alert detail. Catalog has incident_response (ADVANCED_UI).  
   - **Suggestion**: Optional "Incident Response" card in Automation Center (e.g. under Actions or Advanced) that explains it’s per-alert and links to open alerts.

4. **Model Health internals**  
   - Drift and Calibration are run as part of maintenance and separately in Dev. No single "Model Health" report page that shows drift + calibration + conformal in one place (only calibration and redteam reports exist).  
   - **Suggestion**: Either add a "Model Health" report that aggregates drift/calibration/conformal, or document that System Health = model_health agent and link to calibration (and redteam) reports.

---

## 5. What to keep when restructuring

### 5.1 Must keep
- Every component and page listed in §1.
- All API methods and hooks; demo branching and query keys; mutation invalidation lists.
- Demo mode toggle in layout and all demo behaviors in §3.
- Fixtures in `public/fixtures/`.
- Flow: Investigation → run result → trace → artifact links → Actions (outreach) and alert detail (refresh, notify, incident response).

### 5.2 Recommended additions (to tie things together)
1. **Automation Center: "Agents behind the scenes"**  
   - One section or tab that shows: Supervisor = Ingest → Financial → Narrative (optional Ring) → Outreach candidates. Optionally "See pipeline demo" that loads `getFinancialDemo()` and links to Replay or shows a mini trace.

2. **Navigation grouping**  
   - e.g. **Overview**: Today, Alerts; **Evidence**: Graph, Sessions, Watchlists, Rings, Summaries; **Automation**: Automation Center, Reports, Replay; **Data**: Ingest; **Settings**: Settings; **Elder view**.

3. **Incident Response**  
   - Mention in Automation Center (Actions or Advanced) with short copy and link to Alerts.

4. **Financial run from Automation Center**  
   - Optional "Run Financial (demo)" that calls `postFinancialRun({ use_demo_events: true })` or a link "See Financial pipeline" → Replay with API-loaded data.

5. **Single source of truth for "what runs when"**  
   - Use `GET /agents/catalog` (and optionally registry docs) to drive:  
     - Which agents appear where (e.g. DEFAULT_UI vs ADVANCED_UI).  
     - Which actions are available (e.g. run investigation, run maintenance, run incident response per alert).

---

## 6. Checklist before/after restructure

- [ ] All components in §1 still exist and are used.
- [ ] All API client methods and hooks unchanged in contract; demo branching and query keys preserved.
- [ ] Demo toggle in layout; fixture list complete; demo behavior per §3 unchanged.
- [ ] Investigation → run result → trace → artifact links → Actions flow works.
- [ ] Alert detail: Refresh, Notify caregiver, Action Plan (Incident Response), feedback, deep-dive, policy gate.
- [ ] Automation Center: Investigation, Actions, System Health, Advanced (admin) tabs; run maintenance; dev agent buttons.
- [ ] Reports: narrative by id, calibration, redteam; links from Automation Center or Reports hub.
- [ ] Replay: fixture + (when not demo) Load from API (`getFinancialDemo`).
- [ ] Ingest disabled in demo; settings capabilities and consent.
- [ ] (Optional) Automation Center shows "agents behind the scenes" and/or Financial demo.
- [ ] (Optional) Nav grouped for clearer UX.

This document is the single reference for preserving functionality and integrating all coded agents during the frontend restructure.
