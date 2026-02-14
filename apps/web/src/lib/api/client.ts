/**
 * Typed API client for Anchor backend. All data reads go through FastAPI.
 * Uses Zod to validate responses; resilient to missing/partial fields.
 */
import {
  EventsListResponseSchema,
  HouseholdMeSchema,
  RiskSignalDetailSchema,
  RiskSignalListResponseSchema,
  SessionListResponseSchema,
  SimilarIncidentsResponseSchema,
  WatchlistListResponseSchema,
  WeeklySummarySchema,
  type FeedbackLabel,
} from "./schemas";

const getBase = () => {
  if (typeof window !== "undefined") {
    return process.env.NEXT_PUBLIC_API_BASE_URL || "";
  }
  return process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
};

async function request<T>(
  path: string,
  options: RequestInit & { query?: Record<string, string | number | undefined> } = {}
): Promise<T> {
  const { query, ...init } = options;
  const base = getBase();
  const url = new URL(path.startsWith("http") ? path : `${base}${path}`);
  if (query) {
    Object.entries(query).forEach(([k, v]) => {
      if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
    });
  }
  const token = typeof window !== "undefined" ? window.__anchor_token : undefined;
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(init.headers as Record<string, string>),
  };
  if (token) (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;

  const res = await fetch(url.toString(), { ...init, headers });
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const j = JSON.parse(text);
      detail = j.detail ?? text;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  const data = await res.json();
  return data as T;
}

function safeParse<T>(schema: { safeParse: (v: unknown) => { success: boolean; data?: T } }, data: unknown): T {
  const out = schema.safeParse(data);
  if (out.success) return out.data as T;
  console.warn("API schema validation failed", out);
  return data as T;
}

export const api = {
  async getHouseholdMe(): Promise<ReturnType<typeof HouseholdMeSchema.parse>> {
    const data = await request<unknown>("/households/me");
    return safeParse(HouseholdMeSchema, data) as ReturnType<typeof HouseholdMeSchema.parse>;
  },

  async getSessions(params?: { from?: string; to?: string; limit?: number; offset?: number }) {
    const data = await request<unknown>("/sessions", {
      query: {
        from: params?.from,
        to: params?.to,
        limit: params?.limit ?? 50,
        offset: params?.offset ?? 0,
      },
    });
    return safeParse(SessionListResponseSchema, data);
  },

  async getSessionEvents(
    sessionId: string,
    params?: { limit?: number; offset?: number }
  ) {
    const data = await request<unknown>(`/sessions/${sessionId}/events`, {
      query: { limit: params?.limit ?? 50, offset: params?.offset ?? 0 },
    });
    return safeParse(EventsListResponseSchema, data);
  },

  async getRiskSignals(params?: {
    status?: string;
    "severity>="?: number;
    limit?: number;
    offset?: number;
  }) {
    const q: Record<string, string | number | undefined> = {
      limit: params?.limit ?? 50,
      offset: params?.offset ?? 0,
    };
    if (params?.status) q.status = params.status;
    if (params?.["severity>="] != null) q["severity>="] = params["severity>="];
    const data = await request<unknown>("/risk_signals", { query: q });
    return safeParse(RiskSignalListResponseSchema, data);
  },

  async getRiskSignal(id: string) {
    const data = await request<unknown>(`/risk_signals/${id}`);
    return safeParse(RiskSignalDetailSchema, data);
  },

  async getSimilarIncidents(signalId: string, top_k?: number) {
    const data = await request<unknown>(`/risk_signals/${signalId}/similar`, {
      query: { top_k: top_k ?? 5 },
    });
    return safeParse(SimilarIncidentsResponseSchema, data);
  },

  async submitFeedback(signalId: string, body: { label: FeedbackLabel; notes?: string }) {
    const data = await request<{ ok: boolean }>(`/risk_signals/${signalId}/feedback`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    return data;
  },

  async getWatchlists() {
    const data = await request<unknown>("/watchlists");
    return safeParse(WatchlistListResponseSchema, data);
  },

  async getSummaries(params?: { from?: string; to?: string; session_id?: string; limit?: number }) {
    const data = await request<unknown>("/summaries", {
      query: {
        from: params?.from,
        to: params?.to,
        session_id: params?.session_id,
        limit: params?.limit ?? 20,
      },
    });
    if (!Array.isArray(data)) return [];
    return (data as unknown[]).map((d) => safeParse(WeeklySummarySchema, d));
  },
};

export function getWsUrl(path: string): string {
  const base = getBase().replace(/^http/, "ws");
  return `${base}${path}`;
}
