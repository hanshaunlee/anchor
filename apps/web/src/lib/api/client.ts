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
  RiskSignalDetailSubgraphSchema,
  SimilarIncidentsResponseSchema,
  WatchlistListResponseSchema,
  WeeklySummarySchema,
  type FeedbackLabel,
} from "./schemas";

const getBase = () => {
  const base = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
  return base.trim() || "http://localhost:8000";
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

  /** POST /households/onboard — create household and link user after sign-up. Idempotent. */
  async postOnboard(body: { display_name?: string; household_name?: string } = {}) {
    const data = await request<unknown>("/households/onboard", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return safeParse(HouseholdMeSchema, data);
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

  /** GET /agents/status */
  async getAgentsStatus(): Promise<{ agents: { agent_name: string; last_run_at: string | null; last_run_status: string | null; last_run_summary: Record<string, unknown> | null }[] }> {
    return request("/agents/status");
  },

  /** POST /agents/financial/run. dry_run and use_demo_events return risk_signals/watchlists; use_demo_events also returns input_events. */
  async postFinancialRun(body: {
    time_window_days?: number;
    dry_run?: boolean;
    use_demo_events?: boolean;
  }) {
    return request<{
      ok: boolean;
      dry_run: boolean;
      use_demo_events?: boolean;
      run_id?: string;
      risk_signals_count: number;
      watchlists_count: number;
      logs: string[];
      motif_tags?: string[];
      timeline_snippet?: unknown[];
      risk_signals?: unknown[];
      watchlists?: unknown[];
      input_events?: unknown[];
    }>("/agents/financial/run", { method: "POST", body: JSON.stringify(body ?? {}) });
  },

  /** GET /agents/financial/demo — run agent on demo events (no auth). Returns input_events + output for inspection. */
  async getFinancialDemo() {
    return request<{
      ok: boolean;
      message: string;
      input_events: unknown[];
      input_summary: string;
      output: {
        logs: string[];
        motif_tags: string[];
        timeline_snippet: unknown[];
        risk_signals: unknown[];
        watchlists: unknown[];
      };
      risk_signals_count: number;
      watchlists_count: number;
    }>("/agents/financial/demo");
  },

  /** GET /agents/financial/trace?run_id= */
  async getFinancialTrace(runId: string) {
    return request<{
      id: string;
      agent_name: string;
      started_at: string;
      ended_at?: string;
      status?: string;
      summary_json?: Record<string, unknown>;
      step_trace?: Array<{ step?: string; status?: string; error?: string }>;
    }>("/agents/financial/trace", { query: { run_id: runId } });
  },

  /** GET /agents/trace?run_id=&agent_name= — any agent run trace (friendly Agent Trace for UI) */
  async getAgentTrace(runId: string, agentName: string) {
    return request<{
      id: string;
      agent_name: string;
      started_at: string;
      ended_at?: string;
      status?: string;
      summary_json?: Record<string, unknown>;
      step_trace?: Array<{ step?: string; status?: string; error?: string }>;
    }>("/agents/trace", { query: { run_id: runId, agent_name: agentName } });
  },

  /** POST /agents/{slug}/run — run non-financial agent (drift, narrative, ring, calibration, redteam). */
  async postAgentRun(
    slug: "drift" | "narrative" | "ring" | "calibration" | "redteam",
    body: { dry_run?: boolean } = {}
  ) {
    return request<{
      ok: boolean;
      dry_run: boolean;
      run_id?: string | null;
      step_trace?: Array<{ step?: string; status?: string; error?: string }>;
      summary_json?: Record<string, unknown>;
    }>(`/agents/${slug}/run`, { method: "POST", body: JSON.stringify(body ?? { dry_run: true }) });
  },

  /** GET /graph/evidence — household evidence subgraph for Graph view */
  async getGraphEvidence() {
    const data = await request<unknown>("/graph/evidence");
    return safeParse(RiskSignalDetailSubgraphSchema, data);
  },

  /** POST /graph/sync-neo4j — mirror evidence subgraph to Neo4j */
  async postGraphSyncNeo4j() {
    return request<{ ok: boolean; message: string; entities?: number; relationships?: number }>("/graph/sync-neo4j", {
      method: "POST",
    });
  },

  /** GET /graph/neo4j-status — whether Neo4j is configured, browser URL, optional connect_url and password for local */
  async getGraphNeo4jStatus() {
    return request<{ enabled: boolean; browser_url?: string | null; connect_url?: string | null; password?: string | null }>("/graph/neo4j-status");
  },

  /** POST /ingest/events — batch ingest event packets (session/device must belong to household). */
  async postIngestEvents(body: { events: unknown[] }) {
    return request<{ ingested: number; session_ids: string[]; last_ts: string | null }>("/ingest/events", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
};

export function getWsUrl(path: string): string {
  const base = getBase().replace(/^http/, "ws");
  return `${base}${path}`;
}
