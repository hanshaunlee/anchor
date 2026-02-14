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
  score: z.number(),
  outcome: z.string().nullable().optional(),
  ts: iso.nullable().optional(),
});
export type SimilarIncident = z.infer<typeof SimilarIncidentSchema>;

export const SimilarIncidentsResponseSchema = z.object({
  similar: z.array(SimilarIncidentSchema),
});
export type SimilarIncidentsResponse = z.infer<typeof SimilarIncidentsResponseSchema>;

export const WatchlistItemSchema = z.object({
  id: uuid,
  watch_type: z.string(),
  pattern: z.record(z.string(), z.unknown()),
  reason: z.string().nullable(),
  priority: z.number(),
  expires_at: iso.nullable(),
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
