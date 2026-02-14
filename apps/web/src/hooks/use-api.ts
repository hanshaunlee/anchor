"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAppStore } from "@/store/use-app-store";
import { api } from "@/lib/api";
import { HouseholdMeSchema, RiskSignalListResponseSchema, RiskSignalDetailSchema, SessionListResponseSchema, EventsListResponseSchema, WatchlistListResponseSchema, WeeklySummarySchema } from "@/lib/api/schemas";

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
    queryFn: () =>
      demoMode ? Promise.resolve({ similar: [] }) : api.getSimilarIncidents(signalId!, topK),
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
