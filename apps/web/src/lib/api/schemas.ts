/**
 * Zod schemas matching backend api_ui_contracts.md and apps/api/api/schemas.py
 */
import { z } from "zod";

const uuid = z.string().uuid();
const iso = z.string().datetime({ offset: true }).or(z.string());

export const UserRole = z.enum(["elder", "caregiver", "admin"]);
export type UserRole = z.infer<typeof UserRole>;

export const HouseholdMeSchema = z.object({
  id: uuid,
  name: z.string(),
  role: UserRole,
  display_name: z.string().nullable(),
});
export type HouseholdMe = z.infer<typeof HouseholdMeSchema>;

export const CapabilitiesMeSchema = z.object({
  household_id: uuid,
  notify_sms_enabled: z.boolean(),
  notify_email_enabled: z.boolean(),
  device_policy_push_enabled: z.boolean(),
  bank_data_connector: z.string(),
  bank_control_capabilities: z.record(z.string(), z.unknown()),
  updated_at: iso.nullable(),
});
export type CapabilitiesMe = z.infer<typeof CapabilitiesMeSchema>;

export const HouseholdConsentSchema = z.object({
  share_with_caregiver: z.boolean(),
  share_text: z.boolean(),
  allow_outbound_contact: z.boolean(),
  escalation_threshold: z.number(),
  updated_at: iso.nullable().optional(),
});
export type HouseholdConsent = z.infer<typeof HouseholdConsentSchema>;

export const SessionMode = z.enum(["offline", "online"]);
export const SessionListItemSchema = z.object({
  id: uuid,
  device_id: uuid,
  started_at: iso,
  ended_at: iso.nullable(),
  mode: SessionMode,
  consent_state: z.record(z.string(), z.unknown()),
  summary_text: z.string().nullable().optional(),
});
export type SessionListItem = z.infer<typeof SessionListItemSchema>;

export const SessionListResponseSchema = z.object({
  sessions: z.array(SessionListItemSchema),
  total: z.number(),
});
export type SessionListResponse = z.infer<typeof SessionListResponseSchema>;

export const EventListItemSchema = z.object({
  id: uuid,
  ts: iso,
  seq: z.number(),
  event_type: z.string(),
  payload: z.record(z.string(), z.unknown()),
  text_redacted: z.boolean(),
});
export type EventListItem = z.infer<typeof EventListItemSchema>;

export const EventsListResponseSchema = z.object({
  events: z.array(EventListItemSchema),
  total: z.number(),
  next_offset: z.number().nullable(),
});
export type EventsListResponse = z.infer<typeof EventsListResponseSchema>;

export const RiskSignalStatus = z.enum(["open", "acknowledged", "dismissed", "escalated"]);
export type RiskSignalStatus = z.infer<typeof RiskSignalStatus>;

export const RiskSignalCardSchema = z.object({
  id: uuid,
  ts: iso,
  signal_type: z.string(),
  severity: z.number().min(1).max(5),
  score: z.number(),
  status: RiskSignalStatus,
  summary: z.string().nullable().optional(),
  model_available: z.boolean().nullable().optional(),
});
export type RiskSignalCard = z.infer<typeof RiskSignalCardSchema>;

export const SubgraphNodeSchema = z.object({
  id: z.string(),
  type: z.string(),
  label: z.string().nullable().optional(),
  score: z.number().nullable().optional(),
});
export type SubgraphNode = z.infer<typeof SubgraphNodeSchema>;

export const SubgraphEdgeSchema = z.object({
  src: z.string(),
  dst: z.string(),
  type: z.string(),
  weight: z.number().nullable().optional(),
  rank: z.number().nullable().optional(),
});
export type SubgraphEdge = z.infer<typeof SubgraphEdgeSchema>;

export const RiskSignalDetailSubgraphSchema = z.object({
  nodes: z.array(SubgraphNodeSchema),
  edges: z.array(SubgraphEdgeSchema),
});
export type RiskSignalDetailSubgraph = z.infer<typeof RiskSignalDetailSubgraphSchema>;

export const RiskSignalDetailSchema = z.object({
  id: uuid,
  household_id: uuid,
  ts: iso,
  signal_type: z.string(),
  severity: z.number(),
  score: z.number(),
  status: RiskSignalStatus,
  explanation: z.record(z.string(), z.unknown()),
  recommended_action: z.record(z.string(), z.unknown()),
  subgraph: RiskSignalDetailSubgraphSchema.nullable().optional(),
  session_ids: z.array(uuid).optional().default([]),
  event_ids: z.array(uuid).optional().default([]),
  entity_ids: z.array(uuid).optional().default([]),
});
export type RiskSignalDetail = z.infer<typeof RiskSignalDetailSchema>;

export const RiskSignalListResponseSchema = z.object({
  signals: z.array(RiskSignalCardSchema),
  total: z.number(),
});
export type RiskSignalListResponse = z.infer<typeof RiskSignalListResponseSchema>;

export const FeedbackLabel = z.enum(["true_positive", "false_positive", "unsure"]);
export type FeedbackLabel = z.infer<typeof FeedbackLabel>;

export const SimilarIncidentSchema = z.object({
  risk_signal_id: uuid,
  similarity: z.number(),
  score: z.number(),
  ts: iso.nullable().optional(),
  signal_type: z.string().nullable().optional(),
  severity: z.number().nullable().optional(),
  status: z.string().nullable().optional(),
  label_outcome: z.string().nullable().optional(),
  outcome: z.string().nullable().optional(),
});
export type SimilarIncident = z.infer<typeof SimilarIncidentSchema>;

export const RetrievalProvenanceSchema = z.object({
  model_name: z.string().nullable().optional(),
  checkpoint_id: z.string().nullable().optional(),
  embedding_dim: z.number().nullable().optional(),
  timestamp: iso.nullable().optional(),
});
export type RetrievalProvenance = z.infer<typeof RetrievalProvenanceSchema>;

export const SimilarIncidentsResponseSchema = z.object({
  /** When false or omitted, similar incidents are not available (e.g. model not run). Do not default to true so backend value is respected. */
  available: z.boolean().optional(),
  reason: z.string().nullable().optional(),
  similar: z.array(SimilarIncidentSchema),
  retrieval_provenance: RetrievalProvenanceSchema.nullable().optional(),
});
export type SimilarIncidentsResponse = z.infer<typeof SimilarIncidentsResponseSchema>;

/** GET /risk_signals/{id}/page — compound alert detail (composable: detail + page + page_etag). Backend also returns flat keys for compat. */
export const RiskSignalPagePayloadSchema = z.object({
  detail: RiskSignalDetailSchema.optional(),
  page: z
    .object({
      session_events: z.array(EventListItemSchema).default([]),
      similar: SimilarIncidentsResponseSchema,
      actions: z.array(z.record(z.string(), z.unknown())).default([]),
      playbook: z
        .object({
          id: uuid,
          household_id: uuid,
          risk_signal_id: uuid,
          playbook_type: z.string(),
          graph: z.record(z.string(), z.unknown()),
          status: z.string(),
          created_at: iso,
          updated_at: iso,
          tasks: z.array(
            z.object({
              id: uuid,
              playbook_id: uuid,
              task_type: z.string(),
              status: z.string(),
              details: z.record(z.string(), z.unknown()),
              completed_by_user_id: uuid.nullable().optional(),
              completed_at: iso.nullable().optional(),
              created_at: iso,
            })
          ),
        })
        .nullable()
        .optional(),
      gating: z
        .object({
          investigation_refresh_allowed: z.boolean().default(false),
          investigation_refresh_reasons: z.array(z.string()).default([]),
          capabilities_snapshot: z.record(z.string(), z.unknown()).default({}),
        })
        .default({}),
    })
    .optional(),
  page_etag: z.string().nullable().optional(),
  risk_signal_detail: RiskSignalDetailSchema,
  similar_incidents: SimilarIncidentsResponseSchema,
  session_events: z.array(EventListItemSchema).default([]),
  outreach_actions: z.array(z.record(z.string(), z.unknown())).default([]),
  playbook: z
    .object({
      id: uuid,
      household_id: uuid,
      risk_signal_id: uuid,
      playbook_type: z.string(),
      graph: z.record(z.string(), z.unknown()),
      status: z.string(),
      created_at: iso,
      updated_at: iso,
      tasks: z.array(
        z.object({
          id: uuid,
          playbook_id: uuid,
          task_type: z.string(),
          status: z.string(),
          details: z.record(z.string(), z.unknown()),
          completed_by_user_id: uuid.nullable().optional(),
          completed_at: iso.nullable().optional(),
          created_at: iso,
        })
      ),
    })
    .nullable()
    .optional(),
  capabilities_snapshot: z.record(z.string(), z.unknown()).default({}),
  investigation_refresh_allowed: z.boolean().default(false),
  investigation_refresh_reasons: z.array(z.string()).default([]),
});
export type RiskSignalPagePayload = z.infer<typeof RiskSignalPagePayloadSchema>;

export const WatchlistItemSchema = z.object({
  id: uuid,
  watch_type: z.string(),
  pattern: z.record(z.string(), z.unknown()),
  reason: z.string().nullable(),
  priority: z.number(),
  expires_at: iso.nullable(),
  model_available: z.boolean().nullable().optional(),
});
export type WatchlistItem = z.infer<typeof WatchlistItemSchema>;

export const WatchlistListResponseSchema = z.object({
  watchlists: z.array(WatchlistItemSchema),
});
export type WatchlistListResponse = z.infer<typeof WatchlistListResponseSchema>;

export const WeeklySummarySchema = z.object({
  id: uuid,
  period_start: iso.nullable(),
  period_end: iso.nullable(),
  summary_text: z.string(),
  summary_json: z.record(z.string(), z.unknown()),
});
export type WeeklySummary = z.infer<typeof WeeklySummarySchema>;

/** WS message from server */
export const RiskSignalWsMessageSchema = z.object({
  type: z.literal("risk_signal"),
  id: z.string(),
  household_id: z.string(),
  ts: z.string(),
  signal_type: z.string(),
  severity: z.number(),
  score: z.number(),
});
export type RiskSignalWsMessage = z.infer<typeof RiskSignalWsMessageSchema>;

/** Outbound outreach (caregiver notify) */
export const OutreachActionSchema = z.object({
  id: uuid,
  household_id: uuid,
  triggered_by_risk_signal_id: uuid.nullable(),
  action_type: z.string(),
  channel: z.string(),
  recipient_name: z.string().nullable(),
  recipient_contact: z.string().nullable(),
  recipient_contact_last4: z.string().nullable(),
  payload: z.record(z.string(), z.unknown()),
  status: z.enum(["queued", "sent", "delivered", "failed", "suppressed"]),
  provider: z.string(),
  provider_message_id: z.string().nullable(),
  error: z.string().nullable(),
  created_at: iso.nullable(),
  sent_at: iso.nullable(),
  delivered_at: iso.nullable(),
});
export type OutreachAction = z.infer<typeof OutreachActionSchema>;

export const OutreachSummaryCountsSchema = z.object({
  sent: z.number(),
  suppressed: z.number(),
  failed: z.number(),
  queued: z.number(),
  delivered: z.number(),
});
export const OutreachSummaryRecentItemSchema = z.object({
  id: uuid,
  status: z.string(),
  created_at: iso.nullable(),
  sent_at: iso.nullable(),
  error: z.string().nullable(),
  triggered_by_risk_signal_id: uuid.nullable(),
  channel: z.string().nullable(),
  recipient_contact_last4: z.string().nullable(),
});
export const OutreachSummarySchema = z.object({
  counts: OutreachSummaryCountsSchema,
  recent: z.array(OutreachSummaryRecentItemSchema),
});
export type OutreachSummary = z.infer<typeof OutreachSummarySchema>;

// Protection (unified watchlists, rings, reports)
export const ProtectionWatchlistItemSchema = z.object({
  id: z.string(),
  category: z.string(),
  type: z.string(),
  display_label: z.string().nullable(),
  display_value: z.string().nullable(),
  explanation: z.string().nullable(),
  priority: z.number(),
  score: z.number().nullable(),
  source_agent: z.string().nullable(),
  evidence_signal_ids: z.array(z.string()).default([]),
});
export type ProtectionWatchlistItem = z.infer<typeof ProtectionWatchlistItemSchema>;

export const ProtectionWatchlistSummarySchema = z.object({
  total: z.number(),
  by_category: z.record(z.string(), z.number()),
  items: z.array(ProtectionWatchlistItemSchema).default([]),
});
export type ProtectionWatchlistSummary = z.infer<typeof ProtectionWatchlistSummarySchema>;

export const ProtectionRingSummarySchema = z.object({
  id: z.string(),
  household_id: z.string(),
  created_at: z.string(),
  updated_at: z.string(),
  score: z.number(),
  summary_label: z.string().nullable().optional(),
  summary_text: z.string().nullable().optional(),
  members_count: z.number().default(0),
  meta: z.record(z.string(), z.unknown()).default({}),
});
export type ProtectionRingSummary = z.infer<typeof ProtectionRingSummarySchema>;

export const ProtectionReportSummarySchema = z.object({
  kind: z.string(),
  last_run_at: z.string().nullable().optional(),
  last_run_id: z.string().nullable().optional(),
  summary: z.string().nullable().optional(),
  status: z.string().nullable().optional(),
});
export type ProtectionReportSummary = z.infer<typeof ProtectionReportSummarySchema>;

export const ProtectionOverviewSchema = z.object({
  watchlist_summary: ProtectionWatchlistSummarySchema,
  rings_summary: z.array(ProtectionRingSummarySchema).default([]),
  reports_summary: z.array(ProtectionReportSummarySchema).default([]),
  last_updated_at: z.string().nullable().optional(),
  data_freshness: z.record(z.string(), z.unknown()).default({}),
});
export type ProtectionOverview = z.infer<typeof ProtectionOverviewSchema>;

/** GET /protection/summary — counts + previews for dashboard cards */
export const ProtectionSummarySchema = z.object({
  updated_at: z.string().nullable().optional(),
  counts: z.object({ watchlists: z.number(), rings: z.number(), reports: z.number() }).default({ watchlists: 0, rings: 0, reports: 0 }),
  watchlists_preview: z.array(ProtectionWatchlistItemSchema).default([]),
  rings_preview: z.array(ProtectionRingSummarySchema).default([]),
  reports_preview: z.array(ProtectionReportSummarySchema).default([]),
});
export type ProtectionSummary = z.infer<typeof ProtectionSummarySchema>;

/** GET /protection/reports/latest — latest report metadata per type */
export const ProtectionReportsLatestSchema = z.object({
  updated_at: z.string().nullable().optional(),
  reports: z.record(
    z.string(),
    z.object({
      last_run_at: z.string().nullable().optional(),
      last_run_id: z.string().nullable().optional(),
      summary: z.string().nullable().optional(),
      status: z.string().nullable().optional(),
    })
  ),
});
export type ProtectionReportsLatest = z.infer<typeof ProtectionReportsLatestSchema>;
