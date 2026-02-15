"use client";

import { useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAppStore } from "@/store/use-app-store";
import { api } from "@/lib/api";
import { HouseholdMeSchema, CapabilitiesMeSchema, HouseholdConsentSchema, RiskSignalListResponseSchema, RiskSignalDetailSchema, RiskSignalDetailSubgraphSchema, SessionListResponseSchema, EventsListResponseSchema, WatchlistListResponseSchema, WeeklySummarySchema, ProtectionOverviewSchema, ProtectionWatchlistSummarySchema, ProtectionRingSummarySchema, ProtectionReportSummarySchema, ProtectionSummarySchema, ProtectionReportsLatestSchema, type SimilarIncidentsResponse, type CapabilitiesMe, type HouseholdConsent, type RiskSignalPagePayload } from "@/lib/api/schemas";

const HOUSEHOLD_KEY = ["household", "me"] as const;
const SESSIONS_KEY = ["sessions"] as const;
const SESSION_EVENTS_KEY = (id: string) => ["sessions", id, "events"] as const;
const RISK_SIGNALS_KEY = ["risk_signals"] as const;
const RISK_SIGNAL_KEY = (id: string) => ["risk_signals", id] as const;
const RISK_SIGNAL_PAGE_KEY = (id: string) => ["risk_signals", id, "page"] as const;
const SIMILAR_KEY = (id: string) => ["risk_signals", id, "similar"] as const;
const WATCHLISTS_KEY = ["watchlists"] as const;
const RINGS_KEY = ["rings"] as const;
const RING_KEY = (id: string) => ["rings", id] as const;
const SUMMARIES_KEY = ["summaries"] as const;
const PROTECTION_OVERVIEW_KEY = ["protection", "overview"] as const;
const PROTECTION_SUMMARY_KEY = ["protection", "summary"] as const;
const PROTECTION_WATCHLISTS_KEY = ["protection", "watchlists"] as const;
const PROTECTION_RINGS_KEY = ["protection", "rings"] as const;
const PROTECTION_RING_KEY = (id: string) => ["protection", "rings", id] as const;
const PROTECTION_REPORTS_KEY = ["protection", "reports"] as const;
const PROTECTION_REPORTS_LATEST_KEY = ["protection", "reports", "latest"] as const;

async function fetchFixture<T>(path: string, schema: { safeParse: (v: unknown) => { success: boolean; data?: T } }): Promise<T> {
  const base = typeof window !== "undefined" ? "" : process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000";
  const res = await fetch(`${base}${path}`);
  if (!res.ok) throw new Error(`Fixture ${path} failed`);
  const data = await res.json();
  const out = schema.safeParse(data);
  return (out.success ? out.data : data) as T;
}

export function useHouseholdMe() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...HOUSEHOLD_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/household_me.json", HouseholdMeSchema)
        : api.getHouseholdMe(),
    staleTime: 60_000,
  });
}

export function useSessions(params?: { from?: string; to?: string; limit?: number; offset?: number }) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...SESSIONS_KEY, params, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/sessions.json", SessionListResponseSchema)
        : api.getSessions(params),
  });
}

export function useSessionEvents(sessionId: string | null, params?: { limit?: number; offset?: number }) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: sessionId ? [...SESSION_EVENTS_KEY(sessionId), params, demoMode] : ["sessions", "events", "none"],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/session_events.json", EventsListResponseSchema)
        : api.getSessionEvents(sessionId!, params),
    enabled: !!sessionId,
  });
}

export function useRiskSignals(params?: {
  status?: string;
  severityMin?: number;
  limit?: number;
  offset?: number;
}) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...RISK_SIGNALS_KEY, params, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/risk_signals.json", RiskSignalListResponseSchema)
        : api.getRiskSignals({ ...params, "severity>=": params?.severityMin }),
    refetchInterval: demoMode ? false : 30_000,
  });
}

export function useRiskSignalDetail(id: string | null) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: id ? [...RISK_SIGNAL_KEY(id), demoMode] : ["risk_signals", "none"],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/risk_signal_detail.json", RiskSignalDetailSchema)
        : api.getRiskSignal(id!),
    enabled: !!id,
  });
}

/** Single compound request for alert detail page (GET /risk_signals/:id/page). Use this instead of multiple hooks to avoid waterfall. */
export function useAlertPage(id: string | null) {
  const demoMode = useAppStore((s) => s.demoMode);
  const pageCacheRef = useRef<{ data: RiskSignalPagePayload; etag: string | null } | null>(null);
  return useQuery({
    queryKey: id ? [...RISK_SIGNAL_PAGE_KEY(id), demoMode] : ["risk_signals", "page", "none"],
    queryFn: async (): Promise<RiskSignalPagePayload> => {
      if (demoMode) {
        const [detail, similar] = await Promise.all([
          fetchFixture("/fixtures/risk_signal_detail.json", RiskSignalDetailSchema),
          Promise.resolve({ available: true, similar: [] } as SimilarIncidentsResponse),
        ]);
        const payload: RiskSignalPagePayload = {
          risk_signal_detail: detail,
          similar_incidents: similar,
          session_events: [],
          outreach_actions: [],
          playbook: null,
          capabilities_snapshot: {},
          investigation_refresh_allowed: false,
          investigation_refresh_reasons: [],
        };
        return payload;
      }
      const result = await api.getRiskSignalPage(id!, { etag: pageCacheRef.current?.etag ?? undefined });
      if ("notModified" in result && result.notModified) {
        if (pageCacheRef.current?.data) return pageCacheRef.current.data;
        const again = await api.getRiskSignalPage(id!);
        if ("data" in again) {
          pageCacheRef.current = { data: again.data, etag: again.etag };
          return again.data;
        }
        throw new Error("Unexpected 304 on first load");
      }
      pageCacheRef.current = { data: result.data, etag: result.etag };
      return result.data;
    },
    enabled: !!id,
    staleTime: 10_000,
  });
}

/** Mutation: POST /risk_signals/{id}/explain/deep_dive. Invalidates risk signal detail and page on success. */
export function useDeepDiveExplainMutation(signalId: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (mode: "pg" | "gnn" = "pg") => api.postDeepDiveExplain(signalId!, mode),
    onSuccess: () => {
      if (signalId) {
        qc.invalidateQueries({ queryKey: RISK_SIGNAL_KEY(signalId) });
        qc.invalidateQueries({ queryKey: RISK_SIGNAL_PAGE_KEY(signalId) });
      }
    },
  });
}

export function useSimilarIncidents(signalId: string | null, topK?: number) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: signalId ? [...SIMILAR_KEY(signalId), topK, demoMode] : ["similar", "none"],
    queryFn: (): Promise<SimilarIncidentsResponse | null> =>
      demoMode ? Promise.resolve({ available: true, similar: [] }) : api.getSimilarIncidents(signalId!, topK),
    enabled: !!signalId && !demoMode,
  });
}

export function useSubmitFeedback(signalId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { label: "true_positive" | "false_positive" | "unsure"; notes?: string }) =>
      api.submitFeedback(signalId, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: RISK_SIGNAL_KEY(signalId) });
      qc.invalidateQueries({ queryKey: RISK_SIGNAL_PAGE_KEY(signalId) });
      qc.invalidateQueries({ queryKey: RISK_SIGNALS_KEY });
    },
  });
}

export function useWatchlists() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...WATCHLISTS_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/watchlists.json", WatchlistListResponseSchema)
        : api.getWatchlists(),
  });
}

export function useRings() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...RINGS_KEY, demoMode],
    queryFn: () => (demoMode ? Promise.resolve({ rings: [] }) : api.getRings()),
  });
}

export function useRing(id: string | null) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...RING_KEY(id ?? ""), demoMode],
    queryFn: () => (id && !demoMode ? api.getRing(id) : Promise.reject(new Error("No id or demo"))),
    enabled: Boolean(id) && !demoMode,
  });
}

export function useProtectionOverview() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...PROTECTION_OVERVIEW_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/protection_overview.json", ProtectionOverviewSchema)
        : api.getProtectionOverview(),
  });
}

/** GET /protection/summary — counts + previews for dashboard/summary widgets. */
export function useProtectionSummary() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...PROTECTION_SUMMARY_KEY, demoMode],
    queryFn: async () => {
      if (demoMode) {
        const overview = await fetchFixture("/fixtures/protection_overview.json", ProtectionOverviewSchema);
        return ProtectionSummarySchema.parse({
          updated_at: overview.last_updated_at ?? null,
          counts: {
            watchlists: overview.watchlist_summary.total,
            rings: overview.rings_summary.length,
            reports: overview.reports_summary.filter((r) => r.last_run_at).length,
          },
          watchlists_preview: overview.watchlist_summary.items.slice(0, 5),
          rings_preview: overview.rings_summary.slice(0, 5),
          reports_preview: overview.reports_summary,
        });
      }
      return api.getProtectionSummary();
    },
  });
}

/** GET /protection/reports/latest — latest report metadata per type. */
export function useProtectionReportsLatest() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...PROTECTION_REPORTS_LATEST_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? Promise.resolve({ updated_at: null, reports: {} as Record<string, { last_run_at: string | null; last_run_id: string | null; summary: string | null; status: string | null }> })
        : api.getProtectionReportsLatest(),
  });
}

export function useProtectionWatchlists(params?: { category?: string; type?: string; source_agent?: string; limit?: number }) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...PROTECTION_WATCHLISTS_KEY, params, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/protection_watchlists.json", ProtectionWatchlistSummarySchema)
        : api.getProtectionWatchlists(params),
  });
}

export function useProtectionRings() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...PROTECTION_RINGS_KEY, demoMode],
    queryFn: async () => {
      if (demoMode) {
        const data = await fetch(
          (typeof window !== "undefined" ? "" : process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000") + "/fixtures/protection_rings.json"
        ).then((r) => (r.ok ? r.json() : []));
        return Array.isArray(data) ? data.map((d: unknown) => ProtectionRingSummarySchema.parse(d)) : [];
      }
      return api.getProtectionRings();
    },
  });
}

export function useProtectionRing(id: string | null) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: id ? [...PROTECTION_RING_KEY(id), demoMode] : ["protection", "ring", "none"],
    queryFn: () => (id && !demoMode ? api.getProtectionRing(id) : Promise.reject(new Error("No id or demo"))),
    enabled: Boolean(id) && !demoMode,
  });
}

export function useProtectionReports() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...PROTECTION_REPORTS_KEY, demoMode],
    queryFn: async () => {
      if (demoMode) {
        const data = await fetch(
          (typeof window !== "undefined" ? "" : process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000") + "/fixtures/protection_reports.json"
        ).then((r) => (r.ok ? r.json() : []));
        return Array.isArray(data) ? data.map((d: unknown) => ProtectionReportSummarySchema.parse(d)) : [];
      }
      return api.getProtectionReports();
    },
  });
}

export function useSummaries(params?: { from?: string; to?: string; limit?: number }) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...SUMMARIES_KEY, params, demoMode],
    queryFn: async () => {
      if (demoMode) {
        const data = await fetch(
          (typeof window !== "undefined" ? "" : process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000") + "/fixtures/summaries.json"
        ).then((r) => r.json());
        return Array.isArray(data) ? data.map((d: unknown) => WeeklySummarySchema.parse(d)) : [];
      }
      return api.getSummaries(params);
    },
  });
}

const AGENTS_STATUS_KEY = ["agents", "status"] as const;
const AGENTS_CATALOG_KEY = ["agents", "catalog"] as const;
const AGENTS_TRACE_KEY = (runId: string) => ["agents", "financial", "trace", runId] as const;
const INVESTIGATION_KEY = ["investigation"] as const;
const MAINTENANCE_KEY = ["system", "maintenance"] as const;

export function useAgentsStatus() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...AGENTS_STATUS_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? fetch("/fixtures/agents_status.json").then((r) => (r.ok ? r.json() : { agents: [] }))
        : api.getAgentsStatus(),
  });
}

/** GET /agents/catalog — single source of truth for visibility and triggers. */
export function useAgentCatalog() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...AGENTS_CATALOG_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? fetch("/fixtures/agents_catalog.json").then((r) => (r.ok ? r.json() : { catalog: [], role: "elder" }))
        : api.getAgentsCatalog(),
  });
}

/** POST /investigation/run — Run Investigation (supervisor INGEST_PIPELINE). enqueue=true runs on Modal/worker in background. */
export function useInvestigationRunMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { time_window_days?: number; dry_run?: boolean; use_demo_events?: boolean; enqueue?: boolean }) =>
      api.postInvestigationRun(body),
    onSuccess: () => {
      invalidateAfterInvestigation(qc);
    },
  });
}

/** Invalidate all queries that depend on graph/evidence/watchlists/rings/reports after investigation or agent run. */
export function invalidateAfterInvestigation(
  qc: ReturnType<typeof useQueryClient>,
  options?: { riskSignalId?: string }
) {
  qc.invalidateQueries({ queryKey: AGENTS_STATUS_KEY });
  qc.invalidateQueries({ queryKey: RISK_SIGNALS_KEY });
  qc.invalidateQueries({ queryKey: WATCHLISTS_KEY });
  qc.invalidateQueries({ queryKey: OUTREACH_KEY });
  qc.invalidateQueries({ queryKey: GRAPH_EVIDENCE_KEY });
  qc.invalidateQueries({ queryKey: SESSIONS_KEY });
  qc.invalidateQueries({ queryKey: PROTECTION_OVERVIEW_KEY });
  qc.invalidateQueries({ queryKey: PROTECTION_SUMMARY_KEY });
  qc.invalidateQueries({ queryKey: PROTECTION_WATCHLISTS_KEY });
  qc.invalidateQueries({ queryKey: PROTECTION_RINGS_KEY });
  qc.invalidateQueries({ queryKey: PROTECTION_REPORTS_KEY });
  qc.invalidateQueries({ queryKey: PROTECTION_REPORTS_LATEST_KEY });
  if (options?.riskSignalId) {
    qc.invalidateQueries({ queryKey: RISK_SIGNAL_KEY(options.riskSignalId) });
    qc.invalidateQueries({ queryKey: RISK_SIGNAL_PAGE_KEY(options.riskSignalId) });
    qc.invalidateQueries({ queryKey: SIMILAR_KEY(options.riskSignalId) });
  }
}

/** POST /alerts/:id/refresh — supervisor NEW_ALERT for one signal. */
export function useAlertRefreshMutation(alertId: string | null) {
  const qc = useQueryClient();
  const id = alertId;
  return useMutation({
    mutationFn: () => api.postAlertRefresh(id!),
    onSuccess: () => {
      invalidateAfterInvestigation(qc, id ? { riskSignalId: id } : undefined);
    },
  });
}

/** POST /system/maintenance/run — Model Health (admin only). */
export function useMaintenanceRunMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.postMaintenanceRun(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: MAINTENANCE_KEY });
      invalidateAfterInvestigation(qc);
    },
  });
}

/** POST /system/maintenance/clear_risk_signals — Clear all alerts for this household (admin/caregiver). */
export function useClearRiskSignalsMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.postClearRiskSignals(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: RISK_SIGNALS_KEY });
    },
  });
}

/** POST /actions/outreach/preview */
export function useOutreachPreviewMutation() {
  return useMutation({
    mutationFn: (body: { risk_signal_id: string; channel_preference?: string }) =>
      api.postOutreachPreview(body),
  });
}

/** POST /actions/outreach/send */
export function useOutreachSendMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { risk_signal_id: string; channel_preference?: string }) =>
      api.postOutreachSend(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: OUTREACH_KEY });
      qc.invalidateQueries({ queryKey: RISK_SIGNALS_KEY });
      qc.invalidateQueries({ queryKey: AGENTS_STATUS_KEY });
    },
  });
}

/** GET /actions/outreach with optional risk_signal_id (history for one alert). */
export function useOutreachHistory(params?: { risk_signal_id?: string; limit?: number }) {
  return useQuery({
    queryKey: [...OUTREACH_KEY, "history", params],
    queryFn: () => api.getOutreachActions({ risk_signal_id: params?.risk_signal_id, limit: params?.limit ?? 20 }),
  });
}

export function useFinancialRunMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { time_window_days?: number; dry_run?: boolean; use_demo_events?: boolean }) =>
      api.postFinancialRun(body),
    onSuccess: () => {
      invalidateAfterInvestigation(qc);
    },
  });
}

export function useFinancialTrace(runId: string | null) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: runId ? AGENTS_TRACE_KEY(runId) : ["agents", "trace", "none"],
    queryFn: () => api.getFinancialTrace(runId!),
    enabled: !!runId && !demoMode,
  });
}

/** Trace for any agent (use when run_id + agent_name are from a non-financial run). */
export function useAgentTrace(runId: string | null, agentName: string | null) {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: runId && agentName ? ["agents", "trace", runId, agentName] : ["agents", "trace", "none"],
    queryFn: () => api.getAgentTrace(runId!, agentName!),
    enabled: !!runId && !!agentName && !demoMode,
  });
}

const AGENT_SLUG_TO_NAME: Record<string, string> = {
  drift: "graph_drift",
  narrative: "evidence_narrative",
  ring: "ring_discovery",
  calibration: "continual_calibration",
  redteam: "synthetic_redteam",
};

export function useAgentRunMutation(slug: "drift" | "narrative" | "ring" | "calibration" | "redteam") {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { dry_run?: boolean }) => api.postAgentRun(slug, body),
    onSuccess: () => {
      invalidateAfterInvestigation(qc);
    },
  });
}

export { AGENT_SLUG_TO_NAME };

const GRAPH_EVIDENCE_KEY = ["graph", "evidence"] as const;
const GRAPH_NEO4J_STATUS_KEY = ["graph", "neo4j-status"] as const;

export function useGraphEvidence(options?: { liveSync?: boolean }) {
  const liveSync = options?.liveSync ?? false;
  return useQuery({
    queryKey: [...GRAPH_EVIDENCE_KEY, liveSync],
    queryFn: async () => {
      // Always use API so Graph view is informed by Supabase (events → entities/relationships).
      try {
        return await api.getGraphEvidence();
      } catch (e) {
        // Unauthenticated or no household: return empty graph so UI can show "ingest events" etc.
        if (typeof e === "object" && e !== null && "message" in e && /40[13]|no household/i.test(String((e as Error).message)))
          return RiskSignalDetailSubgraphSchema.parse({ nodes: [], edges: [] });
        throw e;
      }
    },
    refetchInterval: liveSync ? 5_000 : false,
  });
}

export function useGraphNeo4jStatus() {
  return useQuery({
    queryKey: GRAPH_NEO4J_STATUS_KEY,
    queryFn: () => api.getGraphNeo4jStatus(),
  });
}

export function useSyncGraphToNeo4jMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.postGraphSyncNeo4j(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: GRAPH_EVIDENCE_KEY });
    },
  });
}

export function useIngestEventsMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { events: unknown[] }) => api.postIngestEvents(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SESSIONS_KEY });
      qc.invalidateQueries({ queryKey: GRAPH_EVIDENCE_KEY });
      qc.invalidateQueries({ queryKey: RISK_SIGNALS_KEY });
    },
  });
}

const OUTREACH_KEY = ["actions", "outreach"] as const;

export function useOutreachMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { risk_signal_id: string; channel_preference?: string; dry_run?: boolean }) =>
      api.postOutreach(body),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: OUTREACH_KEY });
      invalidateAfterInvestigation(qc, { riskSignalId: variables.risk_signal_id });
    },
  });
}

export function useOutreachActions(params?: { limit?: number }) {
  return useQuery({
    queryKey: [...OUTREACH_KEY, params],
    queryFn: () => api.getOutreachActions({ limit: params?.limit ?? 20 }),
  });
}

export function useOutreachSummary(enabled?: boolean) {
  return useQuery({
    queryKey: [...OUTREACH_KEY, "summary"],
    queryFn: () => api.getOutreachSummary(),
    enabled: enabled !== false,
  });
}

/** GET /actions/outreach/candidates — list of outreach candidates with blocking reasons. */
export function useOutreachCandidates(enabled?: boolean) {
  return useQuery({
    queryKey: [...OUTREACH_KEY, "candidates"],
    queryFn: () => api.getOutreachCandidates(),
    enabled: enabled !== false,
  });
}

const PLAYBOOK_KEY = (signalId: string) => ["risk_signals", signalId, "playbook"] as const;
const PLAYBOOK_BY_ID_KEY = (id: string) => ["playbooks", id] as const;
const CAPABILITIES_KEY = ["capabilities", "me"] as const;
const CONSENT_KEY = ["household", "consent"] as const;

export function useRiskSignalPlaybook(signalId: string | null) {
  return useQuery({
    queryKey: signalId ? PLAYBOOK_KEY(signalId) : ["playbook", "none"],
    queryFn: () => api.getRiskSignalPlaybook(signalId!),
    enabled: !!signalId,
  });
}

export function usePlaybook(playbookId: string | null) {
  return useQuery({
    queryKey: playbookId ? PLAYBOOK_BY_ID_KEY(playbookId) : ["playbook", "none"],
    queryFn: () => api.getPlaybook(playbookId!),
    enabled: !!playbookId,
  });
}

export function useCompletePlaybookTaskMutation(playbookId: string, signalId?: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (taskId: string) => api.completePlaybookTask(playbookId, taskId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: PLAYBOOK_BY_ID_KEY(playbookId) });
      if (signalId) {
        qc.invalidateQueries({ queryKey: PLAYBOOK_KEY(signalId) });
        qc.invalidateQueries({ queryKey: RISK_SIGNAL_KEY(signalId) });
        qc.invalidateQueries({ queryKey: RISK_SIGNAL_PAGE_KEY(signalId) });
      }
    },
  });
}

export function useIncidentResponseRunMutation(signalId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { risk_signal_id: string; dry_run?: boolean }) =>
      api.postIncidentResponseRun({ ...body, risk_signal_id: signalId }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: PLAYBOOK_KEY(signalId) });
      qc.invalidateQueries({ queryKey: RISK_SIGNAL_KEY(signalId) });
      qc.invalidateQueries({ queryKey: RISK_SIGNAL_PAGE_KEY(signalId) });
      qc.invalidateQueries({ queryKey: RISK_SIGNALS_KEY });
      qc.invalidateQueries({ queryKey: AGENTS_STATUS_KEY });
    },
  });
}

export function useCapabilitiesMe() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...CAPABILITIES_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/capabilities_me.json", CapabilitiesMeSchema)
        : api.getCapabilitiesMe(),
  });
}

export function usePatchCapabilitiesMutation() {
  const qc = useQueryClient();
  const demoMode = useAppStore((s) => s.demoMode);
  return useMutation({
    mutationFn: async (body: Parameters<typeof api.patchCapabilities>[0]) => {
      if (demoMode) {
        const cur = qc.getQueryData<CapabilitiesMe>([...CAPABILITIES_KEY, true]);
        const base =
          cur ??
          (await fetchFixture("/fixtures/capabilities_me.json", CapabilitiesMeSchema));
        const next: CapabilitiesMe = {
          ...base,
          ...(body.notify_sms_enabled !== undefined && { notify_sms_enabled: body.notify_sms_enabled }),
          ...(body.notify_email_enabled !== undefined && { notify_email_enabled: body.notify_email_enabled }),
          ...(body.device_policy_push_enabled !== undefined && { device_policy_push_enabled: body.device_policy_push_enabled }),
          ...(body.bank_data_connector !== undefined && { bank_data_connector: body.bank_data_connector }),
          bank_control_capabilities: {
            ...(base.bank_control_capabilities as Record<string, unknown>),
            ...(body.bank_control_capabilities || {}),
          },
        };
        qc.setQueryData([...CAPABILITIES_KEY, true], next);
        return next;
      }
      return api.patchCapabilities(body);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: CAPABILITIES_KEY }),
  });
}

export function useHouseholdConsent() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...CONSENT_KEY, demoMode],
    queryFn: () =>
      demoMode
        ? fetchFixture("/fixtures/consent_me.json", HouseholdConsentSchema)
        : api.getHouseholdConsent(),
    staleTime: 60_000,
  });
}

export function usePatchHouseholdConsentMutation() {
  const qc = useQueryClient();
  const demoMode = useAppStore((s) => s.demoMode);
  return useMutation({
    mutationFn: async (body: Parameters<typeof api.patchHouseholdConsent>[0]) => {
      if (demoMode) {
        const cur = qc.getQueryData<HouseholdConsent>([...CONSENT_KEY, true]);
        const base =
          cur ?? (await fetchFixture("/fixtures/consent_me.json", HouseholdConsentSchema));
        const next: HouseholdConsent = { ...base, ...body };
        qc.setQueryData([...CONSENT_KEY, true], next);
        return next;
      }
      return api.patchHouseholdConsent(body);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: CONSENT_KEY }),
  });
}

/** POST /explain — get plain-English explanations for opaque IDs (Claude when configured). */
export function useExplainMutation() {
  return useMutation({
    mutationFn: (body: { context: string; items: Array<{ id: string; hint?: string | null }> }) =>
      api.postExplain(body),
  });
}
