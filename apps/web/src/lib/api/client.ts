/**
 * Typed API client for Anchor backend. All data reads go through FastAPI.
 * Uses Zod to validate responses; resilient to missing/partial fields.
 */
import {
  EventsListResponseSchema,
  HouseholdMeSchema,
  HouseholdConsentSchema,
  RiskSignalDetailSchema,
  RiskSignalListResponseSchema,
  RiskSignalPagePayloadSchema,
  SessionListResponseSchema,
  RiskSignalDetailSubgraphSchema,
  SimilarIncidentsResponseSchema,
  WatchlistListResponseSchema,
  WeeklySummarySchema,
  ProtectionOverviewSchema,
  ProtectionWatchlistSummarySchema,
  ProtectionRingSummarySchema,
  ProtectionReportSummarySchema,
  type FeedbackLabel,
  type RiskSignalPagePayload,
  type ProtectionOverview,
  type ProtectionRingSummary,
  type ProtectionReportSummary,
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

  /** GET /households/me/consent — household consent defaults (outbound, share, etc.). */
  async getHouseholdConsent() {
    const data = await request<unknown>("/households/me/consent");
    return safeParse(HouseholdConsentSchema, data);
  },

  /** PATCH /households/me/consent — update household consent (e.g. allow_outbound_contact). */
  async patchHouseholdConsent(body: {
    share_with_caregiver?: boolean;
    share_text?: boolean;
    allow_outbound_contact?: boolean;
    escalation_threshold?: number;
  }) {
    const data = await request<unknown>("/households/me/consent", {
      method: "PATCH",
      body: JSON.stringify(body),
    });
    return safeParse(HouseholdConsentSchema, data);
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

  /**
   * GET /risk_signals/:id/page — compound alert detail (one round trip).
   * Pass etag for If-None-Match; on 304 returns { notModified: true, etag } and caller keeps cached data.
   */
  async getRiskSignalPage(
    signalId: string,
    opts?: { events_limit?: number; etag?: string }
  ): Promise<{ data: RiskSignalPagePayload; etag: string | null } | { notModified: true; etag: string }> {
    const base = getBase();
    const url = new URL(`${base}/risk_signals/${signalId}/page`);
    if (opts?.events_limit != null) url.searchParams.set("events_limit", String(opts.events_limit));
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    const token = typeof window !== "undefined" ? window.__anchor_token : undefined;
    if (token) headers["Authorization"] = `Bearer ${token}`;
    if (opts?.etag) headers["If-None-Match"] = opts.etag;
    const res = await fetch(url.toString(), { headers });
    const responseEtag = res.headers.get("ETag");
    if (res.status === 304) {
      return { notModified: true, etag: responseEtag ?? opts?.etag ?? "" };
    }
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
    const parsed = safeParse(RiskSignalPagePayloadSchema, data) as RiskSignalPagePayload;
    return { data: parsed, etag: responseEtag };
  },

  async getSimilarIncidents(signalId: string, top_k?: number) {
    const data = await request<unknown>(`/risk_signals/${signalId}/similar`, {
      query: { top_k: top_k ?? 5 },
    });
    return safeParse(SimilarIncidentsResponseSchema, data);
  },

  /** POST /risk_signals/{id}/explain/deep_dive?mode=pg|gnn. Persists deep_dive_subgraph on the risk signal. */
  async postDeepDiveExplain(signalId: string, mode: "pg" | "gnn" = "pg") {
    return request<{ ok: boolean; method: string; deep_dive_subgraph?: { nodes: unknown[]; edges: unknown[] } }>(
      `/risk_signals/${signalId}/explain/deep_dive`,
      { method: "POST", query: { mode } }
    );
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

  /** GET /rings — Ring Discovery agent artifacts. */
  async getRings(): Promise<{ rings: { id: string; household_id: string; created_at: string; updated_at: string; score: number; meta: Record<string, unknown> }[] }> {
    return request("/rings");
  },

  /** GET /rings/:id — ring detail with members. */
  async getRing(ringId: string): Promise<{
    id: string;
    household_id: string;
    created_at: string;
    updated_at: string;
    score: number;
    meta: Record<string, unknown>;
    members: Array<{ entity_id: string | null; role: string | null; first_seen_at: string | null; last_seen_at: string | null }>;
  }> {
    return request(`/rings/${ringId}`);
  },

  /** GET /protection/overview — unified Protection page payload. */
  async getProtectionOverview(): Promise<ProtectionOverview> {
    const data = await request<unknown>("/protection/overview");
    return safeParse(ProtectionOverviewSchema, data) as ProtectionOverview;
  },

  /** GET /protection/watchlists — active watchlist items. */
  async getProtectionWatchlists(params?: { category?: string; type?: string; source_agent?: string; limit?: number }) {
    const data = await request<unknown>("/protection/watchlists", { query: params });
    return safeParse(ProtectionWatchlistSummarySchema, data);
  },

  /** GET /protection/rings — active rings. */
  async getProtectionRings(): Promise<ProtectionRingSummary[]> {
    const data = await request<unknown>("/protection/rings");
    return Array.isArray(data) ? (data as unknown[]).map((d) => safeParse(ProtectionRingSummarySchema, d) as ProtectionRingSummary) : [];
  },

  /** GET /protection/rings/:id — ring detail. */
  async getProtectionRing(ringId: string) {
    return request<{
      id: string;
      household_id: string;
      created_at: string;
      updated_at: string;
      score: number;
      meta: Record<string, unknown>;
      members: Array<{ entity_id: string | null; role: string | null; first_seen_at: string | null; last_seen_at: string | null }>;
      summary_label?: string;
      summary_text?: string;
    }>(`/protection/rings/${ringId}`);
  },

  /** GET /protection/reports — report summaries. */
  async getProtectionReports(): Promise<ProtectionReportSummary[]> {
    const data = await request<unknown>("/protection/reports");
    return Array.isArray(data) ? (data as unknown[]).map((d) => safeParse(ProtectionReportSummarySchema, d) as ProtectionReportSummary) : [];
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
  async getAgentsStatus(): Promise<{ agents: { agent_name: string; last_run_id: string | null; last_run_at: string | null; last_run_status: string | null; last_run_summary: Record<string, unknown> | null }[] }> {
    return request("/agents/status");
  },

  /** GET /agents/catalog — registry filtered by role, consent, env, calibration, model. Single source of truth for visibility. */
  async getAgentsCatalog() {
    return request<{
      catalog: Array<{
        agent_name: string;
        slug: string;
        label: string;
        display_name: string;
        tier: string;
        triggers: string[];
        visibility: string;
        required_roles: string[];
        consent_requirements: string[];
        runnable: boolean;
        reason: string | null;
        recommended?: boolean;
      }>;
      role: string;
    }>("/agents/catalog");
  },

  /** POST /investigation/run — supervisor INGEST_PIPELINE. Caregiver/admin only. enqueue=true runs on Modal/worker in background. */
  async postInvestigationRun(body: { time_window_days?: number; dry_run?: boolean; use_demo_events?: boolean; enqueue?: boolean } = {}) {
    return request<{
      ok: boolean;
      supervisor_run_id: string | null;
      mode: string;
      child_run_ids: Record<string, string | null>;
      created_signal_ids: string[];
      updated_signal_ids: string[];
      created_watchlist_ids: string[];
      outreach_candidates: Array<{ risk_signal_id?: string; severity?: number; calibrated_p?: number; fusion_score?: number; decision_rule_used?: string }>;
      summary_json: { counts?: Record<string, number>; thresholds_used?: Record<string, unknown>; warnings?: string[] };
      step_trace: Array<{ step?: string; status?: string; error?: string; notes?: string; outputs_count?: number; started_at?: string; ended_at?: string }>;
      warnings: string[];
    }>("/investigation/run", { method: "POST", body: JSON.stringify(body ?? {}) });
  },

  /** POST /alerts/:id/refresh — supervisor NEW_ALERT for one signal. Caregiver/admin only. */
  async postAlertRefresh(alertId: string) {
    return request<{
      ok: boolean;
      supervisor_run_id: string | null;
      mode: string;
      child_run_ids: Record<string, string | null>;
      summary_json: Record<string, unknown>;
      step_trace: Array<{ step?: string; status?: string }>;
    }>(`/alerts/${alertId}/refresh`, { method: "POST" });
  },

  /** POST /system/maintenance/run — NIGHTLY_MAINTENANCE (model_health). Admin only. */
  async postMaintenanceRun() {
    return request<{
      ok: boolean;
      supervisor_run_id: string | null;
      mode: string;
      child_run_ids: Record<string, string | null>;
      summary_json: Record<string, unknown>;
      step_trace: Array<{ step?: string; status?: string }>;
    }>("/system/maintenance/run", { method: "POST" });
  },

  /** POST /actions/outreach/preview — preview drafts (caregiver_full, elder_safe). Caregiver/admin only. */
  async postOutreachPreview(body: { risk_signal_id: string; channel_preference?: string }) {
    return request<{
      ok: boolean;
      preview: {
        caregiver_full?: string;
        elder_safe?: string;
        evidence_bundle_summary?: unknown;
        calibrated_score_context?: unknown;
        step_trace?: unknown[];
      };
      suppressed: boolean;
    }>("/actions/outreach/preview", { method: "POST", body: JSON.stringify(body) });
  },

  /** POST /actions/outreach/send — execute send. Caregiver/admin only. */
  async postOutreachSend(body: { risk_signal_id: string; channel_preference?: string }) {
    return request<{
      ok: boolean;
      outbound_action: Record<string, unknown> | null;
      agent_run_id: string | null;
      sent: boolean;
      suppressed: boolean;
    }>("/actions/outreach/send", { method: "POST", body: JSON.stringify(body) });
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

  /** GET /agents/financial/demo — run agent on demo events (no auth). Returns input_events + output + step_trace for replay. */
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
        step_trace?: Array<{ step?: string; status?: string; started_at?: string; ended_at?: string; notes?: string; outputs_count?: number }>;
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

  /** GET /agents/calibration/report — latest Continual Calibration run summary. */
  async getCalibrationReport(): Promise<{ run_id: string; started_at: string; summary_json: Record<string, unknown> }> {
    return request("/agents/calibration/report");
  },

  /** GET /agents/redteam/report — latest Synthetic Red-Team run (includes replay_payload for Open in replay). */
  async getRedteamReport(): Promise<{ run_id: string; started_at: string; summary_json: Record<string, unknown> }> {
    return request("/agents/redteam/report");
  },

  /** GET /agents/narrative/report/:id — Evidence Narrative persisted report. */
  async getNarrativeReport(reportId: string) {
    return request<{
      id: string;
      household_id: string;
      agent_run_id: string | null;
      risk_signal_ids: string[];
      report_json: {
        headline?: string;
        narrative_preview?: string;
        reports?: Array<{
          signal_id: string;
          narrative_text?: string;
          caregiver_narrative?: { headline?: string; summary?: string; key_evidence_bullets?: string[]; recommended_next_steps?: string[] };
          elder_safe?: { plain_language_summary?: string; do_now_checklist?: string[]; reassurance_line?: string };
          hypotheses?: unknown[];
          what_changed?: Record<string, unknown>;
        }>;
      };
      created_at: string;
    }>(`/agents/narrative/report/${reportId}`);
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

  /** POST /actions/outreach — trigger caregiver outreach (caregiver/admin only). */
  async postOutreach(body: { risk_signal_id: string; channel_preference?: string; dry_run?: boolean }) {
    return request<{
      ok: boolean;
      outbound_action: Record<string, unknown> | null;
      agent_run_id: string | null;
      preview?: { caregiver_message?: string; step_trace?: unknown[] };
      suppressed: boolean;
      sent: boolean;
    }>("/actions/outreach", { method: "POST", body: JSON.stringify(body) });
  },

  /** GET /actions/outreach — list recent outbound actions for household. Optional risk_signal_id filter. */
  async getOutreachActions(params?: { household_id?: string; risk_signal_id?: string; limit?: number }) {
    const data = await request<{ actions: Record<string, unknown>[] }>("/actions/outreach", {
      query: { household_id: params?.household_id, risk_signal_id: params?.risk_signal_id, limit: params?.limit ?? 20 },
    });
    return data;
  },

  /** GET /actions/outreach/:id — get one outbound action (elder sees elder_safe only). */
  async getOutreachAction(id: string) {
    return request<Record<string, unknown>>(`/actions/outreach/${id}`);
  },

  /** GET /actions/outreach/candidates — queued outreach candidates with blocking reasons. Caregiver/admin only. */
  async getOutreachCandidates() {
    return request<{
      candidates: Array<{
        risk_signal_id: string;
        outbound_action_id?: string;
        severity?: number;
        signal_type?: string;
        created_at?: string;
        candidate_reason?: string;
        consent_ok: boolean;
        missing_consent_keys: string[];
        caregiver_contact_present: boolean;
        blocking_reasons: string[];
        draft_available?: boolean;
      }>;
    }>("/actions/outreach/candidates");
  },

  /** GET /actions/outreach/summary — counts (sent/suppressed/failed) + recent. Caregiver/admin only. */
  async getOutreachSummary() {
    return request<{ counts: { sent: number; suppressed: number; failed: number; queued: number; delivered: number }; recent: Array<{ id: string; status: string; created_at: string | null; sent_at: string | null; error: string | null; triggered_by_risk_signal_id: string | null; channel: string | null; recipient_contact_last4: string | null }> }>("/actions/outreach/summary");
  },

  /** POST /agents/outreach/run — run caregiver outreach agent (caregiver/admin only). */
  async postOutreachRun(body: { risk_signal_id: string; dry_run?: boolean }) {
    return request<{
      ok: boolean;
      dry_run: boolean;
      run_id?: string | null;
      outbound_action_id?: string | null;
      step_trace?: Array<{ step?: string; status?: string; error?: string }>;
      summary_json?: Record<string, unknown>;
      suppressed: boolean;
      sent: boolean;
    }>("/agents/outreach/run", { method: "POST", body: JSON.stringify(body) });
  },

  /** GET /risk_signals/:id/playbook — playbook for this signal. */
  async getRiskSignalPlaybook(signalId: string) {
    return request<{
      id: string;
      household_id: string;
      risk_signal_id: string;
      playbook_type: string;
      graph: { nodes: unknown[]; edges: unknown[] };
      status: string;
      created_at: string;
      updated_at: string;
      tasks: Array<{
        id: string;
        playbook_id: string;
        task_type: string;
        status: string;
        details: Record<string, unknown>;
        completed_by_user_id: string | null;
        completed_at: string | null;
        created_at: string;
      }>;
    }>(`/risk_signals/${signalId}/playbook`);
  },

  /** GET /playbooks/:id */
  async getPlaybook(playbookId: string) {
    return request<{
      id: string;
      household_id: string;
      risk_signal_id: string;
      playbook_type: string;
      graph: { nodes: unknown[]; edges: unknown[] };
      status: string;
      created_at: string;
      updated_at: string;
      tasks: Array<{
        id: string;
        playbook_id: string;
        task_type: string;
        status: string;
        details: Record<string, unknown>;
        completed_by_user_id: string | null;
        completed_at: string | null;
        created_at: string;
      }>;
    }>(`/playbooks/${playbookId}`);
  },

  /** POST /playbooks/:playbookId/tasks/:taskId/complete */
  async completePlaybookTask(playbookId: string, taskId: string, body?: { notes?: string }) {
    return request<{ ok: boolean; task_id: string; status: string }>(
      `/playbooks/${playbookId}/tasks/${taskId}/complete`,
      { method: "POST", body: JSON.stringify(body ?? {}) }
    );
  },

  /** GET /incident_packets/:id */
  async getIncidentPacket(packetId: string) {
    return request<{ id: string; household_id: string; risk_signal_id: string; packet_json: Record<string, unknown>; created_at: string }>(
      `/incident_packets/${packetId}`
    );
  },

  /** POST /agents/incident-response/run */
  async postIncidentResponseRun(body: { risk_signal_id: string; dry_run?: boolean }) {
    return request<{
      ok: boolean;
      dry_run: boolean;
      run_id?: string | null;
      playbook_id?: string | null;
      incident_packet_id?: string | null;
      step_trace?: Array<{ step?: string; status?: string; error?: string }>;
      summary_json?: Record<string, unknown>;
    }>("/agents/incident-response/run", { method: "POST", body: JSON.stringify(body) });
  },

  /** GET /capabilities/me */
  async getCapabilitiesMe() {
    return request<{
      household_id: string;
      notify_sms_enabled: boolean;
      notify_email_enabled: boolean;
      device_policy_push_enabled: boolean;
      bank_data_connector: string;
      bank_control_capabilities: Record<string, unknown>;
      updated_at: string | null;
    }>("/capabilities/me");
  },

  /** PATCH /capabilities */
  async patchCapabilities(body: {
    notify_sms_enabled?: boolean;
    notify_email_enabled?: boolean;
    device_policy_push_enabled?: boolean;
    bank_data_connector?: string;
    bank_control_capabilities?: Record<string, unknown>;
  }) {
    return request<{
      household_id: string;
      notify_sms_enabled: boolean;
      notify_email_enabled: boolean;
      device_policy_push_enabled: boolean;
      bank_data_connector: string;
      bank_control_capabilities: Record<string, unknown>;
      updated_at: string | null;
    }>("/capabilities", { method: "PATCH", body: JSON.stringify(body) });
  },
};

export function getWsUrl(path: string): string {
  const base = getBase().replace(/^http/, "ws");
  return `${base}${path}`;
}
