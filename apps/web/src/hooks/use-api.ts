"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAppStore } from "@/store/use-app-store";
import { api } from "@/lib/api";
import { HouseholdMeSchema, RiskSignalListResponseSchema, RiskSignalDetailSchema, RiskSignalDetailSubgraphSchema, SessionListResponseSchema, EventsListResponseSchema, WatchlistListResponseSchema, WeeklySummarySchema, type SimilarIncidentsResponse } from "@/lib/api/schemas";

const HOUSEHOLD_KEY = ["household", "me"] as const;
const SESSIONS_KEY = ["sessions"] as const;
const SESSION_EVENTS_KEY = (id: string) => ["sessions", id, "events"] as const;
const RISK_SIGNALS_KEY = ["risk_signals"] as const;
const RISK_SIGNAL_KEY = (id: string) => ["risk_signals", id] as const;
const SIMILAR_KEY = (id: string) => ["risk_signals", id, "similar"] as const;
const WATCHLISTS_KEY = ["watchlists"] as const;
const SUMMARIES_KEY = ["summaries"] as const;

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
const AGENTS_TRACE_KEY = (runId: string) => ["agents", "financial", "trace", runId] as const;

export function useAgentsStatus() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...AGENTS_STATUS_KEY, demoMode],
    queryFn: () => api.getAgentsStatus(),
    enabled: !demoMode,
  });
}

export function useFinancialRunMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { time_window_days?: number; dry_run?: boolean; use_demo_events?: boolean }) =>
      api.postFinancialRun(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: AGENTS_STATUS_KEY });
      qc.invalidateQueries({ queryKey: RISK_SIGNALS_KEY });
      qc.invalidateQueries({ queryKey: WATCHLISTS_KEY });
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
      qc.invalidateQueries({ queryKey: AGENTS_STATUS_KEY });
    },
  });
}

export { AGENT_SLUG_TO_NAME };

const GRAPH_EVIDENCE_KEY = ["graph", "evidence"] as const;
const GRAPH_NEO4J_STATUS_KEY = ["graph", "neo4j-status"] as const;

export function useGraphEvidence() {
  const demoMode = useAppStore((s) => s.demoMode);
  return useQuery({
    queryKey: [...GRAPH_EVIDENCE_KEY, demoMode],
    queryFn: async () => {
      if (demoMode) {
        const data = await fetch(
          (typeof window !== "undefined" ? "" : process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000") + "/fixtures/graph_evidence.json"
        ).then((r) => r.ok ? r.json() : { nodes: [], edges: [] });
        return RiskSignalDetailSubgraphSchema.parse(data);
      }
      return api.getGraphEvidence();
    },
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
    },
  });
}
