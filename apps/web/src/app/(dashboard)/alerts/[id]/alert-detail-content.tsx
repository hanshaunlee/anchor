"use client";

import React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  useRiskSignalDetail,
  useSessionEvents,
  useSimilarIncidents,
  useSubmitFeedback,
  useHouseholdMe,
  useOutreachMutation,
  useOutreachActions,
  useDeepDiveExplainMutation,
  useRiskSignalPlaybook,
  useIncidentResponseRunMutation,
  useCompletePlaybookTaskMutation,
  useCapabilitiesMe,
} from "@/hooks/use-api";
import { GraphEvidence } from "@/components/graph-evidence";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
import { PolicyGateCard } from "@/components/policy-gate-card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { ArrowLeft, Bell, ClipboardCopy, CheckCircle, ShieldAlert, FileJson } from "lucide-react";

const statusVariant: Record<string, string> = {
  open: "bg-status-open/15 text-status-open border-status-open/30",
  acknowledged: "bg-status-acknowledged/15 text-status-acknowledged border-status-acknowledged/30",
  dismissed: "bg-status-dismissed/15 text-status-dismissed border-status-dismissed/30",
  escalated: "bg-status-escalated/15 text-status-escalated border-status-escalated/30",
};

const defaultTraceSteps: TraceStep[] = [
  { step: "Ingest", description: "Loaded events from session.", inputs: "session_ids", outputs: "events", status: "success", latency_ms: 12 },
  { step: "Normalize", description: "Extracted utterances, entities, mentions.", outputs: "utterances, entities", status: "success", latency_ms: 45 },
  { step: "GraphUpdate", description: "Updated household graph.", status: "success", latency_ms: 28 },
  { step: "Score", description: "Ran risk model.", outputs: "risk_score", status: "success", latency_ms: 120 },
  { step: "Explain", description: "Generated motifs and evidence subgraph.", outputs: "motifs, subgraph", status: "success", latency_ms: 85 },
  { step: "ConsentGate", description: "Checked consent for escalation and watchlist.", status: "success", latency_ms: 2 },
  { step: "Watchlist", description: "Synthesized watchlist entries.", status: "success", latency_ms: 15 },
  { step: "EscalationDraft", description: "Drafted caregiver notification (not sent).", status: "success", latency_ms: 8 },
  { step: "Persist", description: "Saved risk signal and watchlist.", status: "success", latency_ms: 35 },
];

export function AlertDetailContent({ id }: { id: string }) {
  const { data: me } = useHouseholdMe();
  const { data: detail, isLoading } = useRiskSignalDetail(id);
  const sessionId = detail?.session_ids?.[0];
  const { data: eventsData } = useSessionEvents(sessionId ?? null, { limit: 10 });
  const { data: similarData } = useSimilarIncidents(id, 5);
  const submitFeedback = useSubmitFeedback(id);
  const outreachMutation = useOutreachMutation();
  const deepDiveMutation = useDeepDiveExplainMutation(id);
  const [graphView, setGraphView] = React.useState<"model" | "deep_dive">("model");
  const { data: outreachData } = useOutreachActions({ limit: 30 });
  const [notes, setNotes] = React.useState("");
  const [outreachPreview, setOutreachPreview] = React.useState<{ caregiver_message?: string; elder_safe_message?: string } | null>(null);
  const [outreachConfirm, setOutreachConfirm] = React.useState(false);
  const events = eventsData?.events ?? [];
  const canTriggerOutreach = me?.role === "caregiver" || me?.role === "admin";
  const outreachForThisSignal = (outreachData?.actions ?? []).filter(
    (a: Record<string, unknown>) => String(a.triggered_by_risk_signal_id) === id
  );
  const { data: playbookData, isLoading: playbookLoading, error: playbookError } = useRiskSignalPlaybook(id);
  const incidentResponseMutation = useIncidentResponseRunMutation(id);
  const completeTaskMutation = useCompletePlaybookTaskMutation(playbookData?.id ?? "", id);
  const { data: capabilities } = useCapabilitiesMe();
  const [copiedScript, setCopiedScript] = React.useState(false);
  const playbook = playbookData;
  const hasLockCardCap = capabilities?.bank_control_capabilities?.lock_card === true;

  if (isLoading || !detail) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  const expl = detail.explanation as Record<string, unknown>;
  const modelAvailable = expl.model_available as boolean | undefined;
  const redacted = expl.redacted as boolean | undefined;
  const redactionReason = expl.redaction_reason as string | undefined;
  const evidenceQuality = expl.model_evidence_quality as { sparsity?: number; edges_kept?: number; edges_total?: number } | undefined;
  const motifs = (expl.motif_tags as string[]) ?? (expl.motifs as string[]) ?? [];
  const recommended = (detail.recommended_action as Record<string, unknown>) ?? {};
  const steps = (recommended.checklist as string[]) ?? (recommended.steps as string[]) ?? [];
  const traceSteps = (expl.logs as TraceStep[] | undefined)
    ? (expl.logs as TraceStep[]).map((l) => ({
        ...l,
        status: (l.status as TraceStep["status"]) ?? "success",
      }))
    : defaultTraceSteps;

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-4">
        <Link href="/alerts" className="rounded-xl p-2 hover:bg-accent">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-semibold tracking-tight truncate">
            Risk signal {id.slice(0, 8)}…
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            {new Date(detail.ts).toLocaleString()} · {detail.signal_type}
          </p>
        </div>
        <Badge className={cn("rounded-lg border", statusVariant[detail.status] ?? "bg-muted")}>
          {detail.status}
        </Badge>
        <span className="rounded-lg bg-destructive/15 px-2 py-1 text-xs font-medium text-destructive">
          Severity {detail.severity}
        </span>
        {detail.signal_type === "ring_candidate" && (expl.ring_id as string) && (
          <Link href={`/rings/${expl.ring_id}`}>
            <span className="rounded-lg bg-amber-500/15 px-2 py-1 text-xs font-medium text-amber-700 dark:text-amber-400 border border-amber-500/30 hover:bg-amber-500/25 cursor-pointer inline-block">
              View ring
            </span>
          </Link>
        )}
        {detail.signal_type === "drift_warning" && (
          <span className="rounded-lg bg-orange-500/15 px-2 py-1 text-xs font-medium text-orange-700 dark:text-orange-400 border border-orange-500/30">
            Drift warning
          </span>
        )}
        {detail.signal_type === "watchlist_embedding_match" && (
          <span className="rounded-lg bg-violet-500/15 px-2 py-1 text-xs font-medium text-violet-700 dark:text-violet-400 border border-violet-500/30">
            Matched centroid watchlist
          </span>
        )}
        {modelAvailable === true && (
          <span className="rounded-lg bg-primary/15 px-2 py-1 text-xs font-medium text-primary border border-primary/30">
            GNN
          </span>
        )}
        {modelAvailable === false && (
          <span className="rounded-lg bg-muted px-2 py-1 text-xs font-medium text-muted-foreground">
            Model unavailable
          </span>
        )}
      </div>

      {redacted && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-800 dark:text-amber-200">
          {redactionReason ?? "Some content is redacted due to consent settings. Raw utterance text and entity labels are withheld."}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Summary</CardTitle>
          </CardHeader>
          <CardContent>
            {(expl.narrative as string) ? (
              <p className="text-sm">{(expl.narrative as string)}</p>
            ) : (
              <p className="text-sm">{(expl.summary as string) ?? "—"}</p>
            )}
            {expl.narrative_evidence_only === true && (
              <span className="inline-block mt-1 rounded-md bg-primary/15 px-2 py-0.5 text-xs font-medium text-primary border border-primary/30">
                Evidence-only
              </span>
            )}
            <p className="text-muted-foreground text-sm mt-2">
              Score: {(detail.score * 100).toFixed(0)}%
            </p>
            {evidenceQuality != null && (
              <p className="text-muted-foreground text-xs mt-2">
                Evidence: {evidenceQuality.edges_kept ?? 0} / {evidenceQuality.edges_total ?? 0} edges
                {typeof evidenceQuality.sparsity === "number" && ` · sparsity ${(evidenceQuality.sparsity * 100).toFixed(0)}%`}
              </p>
            )}
          </CardContent>
        </Card>
        {(detail.signal_type === "watchlist_embedding_match" || (expl.watchlist_match as Record<string, unknown>)) && (
          <Card className="rounded-2xl shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Matched centroid watchlist</CardTitle>
              <p className="text-muted-foreground text-sm">This signal matched a GNN embedding centroid watchlist.</p>
            </CardHeader>
            <CardContent>
              {(() => {
                const wm = (expl.watchlist_match ?? expl) as { similarity?: number; threshold?: number; watchlist_id?: string; centroid_version?: string };
                const sim = wm.similarity ?? detail.score;
                const thresh = wm.threshold;
                return (
                  <p className="text-sm">
                    Similarity {(typeof sim === "number" ? sim * 100 : Number(sim) * 100).toFixed(0)}%
                    {thresh != null && ` (threshold ${(thresh * 100).toFixed(0)}%)`}
                    {wm.centroid_version && ` · ${wm.centroid_version}`}
                  </p>
                );
              })()}
            </CardContent>
          </Card>
        )}
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Recommended action</CardTitle>
          </CardHeader>
          <CardContent>
            {Array.isArray(steps) && steps.length > 0 ? (
              <ul className="list-disc list-inside space-y-1 text-sm">
                {steps.map((s, i) => (
                  <li key={i}>{String(s)}</li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground text-sm">No specific steps.</p>
            )}
          </CardContent>
        </Card>
      </div>

      {motifs.length > 0 && (
        <div className="flex flex-wrap gap-2">
          <span className="text-muted-foreground text-sm font-medium">Motifs:</span>
          {motifs.map((m, i) => (
            <Badge key={i} variant="secondary" className="rounded-lg">
              {String(m)}
            </Badge>
          ))}
        </div>
      )}

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Timeline (key events)</CardTitle>
          <p className="text-muted-foreground text-sm">Events referenced by evidence</p>
        </CardHeader>
        <CardContent>
          {events.length === 0 ? (
            <p className="text-muted-foreground text-sm">No events loaded or redacted.</p>
          ) : (
            <ul className="space-y-3">
              {events.slice(0, 5).map((e) => (
                <li
                  key={e.id}
                  className="flex gap-3 rounded-xl border border-border p-3 text-sm"
                >
                  <span className="text-muted-foreground shrink-0">
                    {new Date(e.ts).toLocaleTimeString()}
                  </span>
                  <span className="font-medium">{e.event_type}</span>
                  {e.text_redacted ? (
                    <span className="text-muted-foreground italic">Redacted due to consent</span>
                  ) : (
                    <span className="truncate">
                      {typeof (e.payload as Record<string, unknown>)?.text === "string"
                        ? (e.payload as Record<string, unknown>).text as string
                        : JSON.stringify(e.payload).slice(0, 60)}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      {detail.subgraph && (detail.subgraph.nodes.length > 0 || detail.subgraph.edges.length > 0) && (
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle className="text-base">Graph evidence</CardTitle>
                <p className="text-muted-foreground text-sm">
                  {modelAvailable === false
                    ? "Rule/motif evidence only (model unavailable)."
                    : "Entity and relationship evidence (edge thickness = importance)."}
                </p>
              </div>
              {modelAvailable === true && (
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground text-xs font-medium">View:</span>
                  <button
                    type="button"
                    onClick={() => setGraphView("model")}
                    className={cn(
                      "rounded-lg px-2 py-1 text-xs font-medium",
                      graphView === "model" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                    )}
                  >
                    PGExplainer
                  </button>
                  <button
                    type="button"
                    onClick={() => setGraphView("deep_dive")}
                    className={cn(
                      "rounded-lg px-2 py-1 text-xs font-medium",
                      graphView === "deep_dive" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                    )}
                  >
                    Deep dive
                  </button>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {graphView === "deep_dive" && !(expl.deep_dive_subgraph as Record<string, unknown>)?.nodes?.length && !(expl.deep_dive_subgraph as Record<string, unknown>)?.edges?.length ? (
              <div className="space-y-3 py-4">
                <p className="text-muted-foreground text-sm">Deep dive subgraph not computed yet.</p>
                <Button
                  size="sm"
                  variant="secondary"
                  className="rounded-xl"
                  disabled={deepDiveMutation.isPending}
                  onClick={() => deepDiveMutation.mutate("pg")}
                >
                  {deepDiveMutation.isPending ? "Computing…" : "Compute deep dive"}
                </Button>
                {deepDiveMutation.isError && (
                  <p className="text-destructive text-sm">{String(deepDiveMutation.error?.message ?? deepDiveMutation.error)}</p>
                )}
              </div>
            ) : graphView === "deep_dive" && (expl.deep_dive_subgraph as { nodes?: unknown[]; edges?: unknown[] })?.nodes ? (
              <GraphEvidence
                variant="replay"
                nodes={(expl.deep_dive_subgraph as { nodes: { id: string; type?: string; label?: string | null; score?: number | null }[] }).nodes.map((n) => ({ id: String(n.id), type: n.type ?? "entity", label: n.label ?? null, score: n.score ?? null }))}
                edges={((expl.deep_dive_subgraph as { edges?: { src: string; dst: string; type?: string; weight?: number; importance?: number; rank?: number }[] }).edges ?? []).map((e) => ({ src: e.src, dst: e.dst, type: e.type ?? "", weight: e.weight ?? e.importance ?? null, rank: e.rank ?? null }))}
                onPlayPath={() => {}}
              />
            ) : (
              <GraphEvidence
                variant="replay"
                nodes={detail.subgraph!.nodes}
                edges={detail.subgraph!.edges}
                onPlayPath={() => {}}
              />
            )}
          </CardContent>
        </Card>
      )}

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Similar incidents</CardTitle>
          <p className="text-muted-foreground text-sm">
            {similarData?.available !== true
              ? (similarData?.reason === "model_not_run" ? "Unavailable (model not run)." : "Unavailable.")
              : "Past signals with similar patterns"}
          </p>
        </CardHeader>
        <CardContent>
          {similarData?.available !== true ? (
            <p className="text-muted-foreground text-sm">Similar incidents require GNN embeddings; none stored for this signal.</p>
          ) : similarData?.similar && similarData.similar.length > 0 ? (
            <>
              <ul className="space-y-2">
                {similarData.similar.map((s) => (
                  <li key={s.risk_signal_id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border px-3 py-2 text-sm">
                    <Link href={`/alerts/${s.risk_signal_id}`} className="hover:underline font-medium">
                      {String(s.risk_signal_id).slice(0, 8)}…
                    </Link>
                    <span className="text-muted-foreground">
                      similarity {((s.similarity ?? s.score) * 100).toFixed(0)}%
                      {(s.label_outcome ?? s.outcome) && ` · ${s.label_outcome ?? s.outcome}`}
                      {s.severity != null && ` · severity ${s.severity}`}
                    </span>
                  </li>
                ))}
              </ul>
              {similarData.retrieval_provenance && (
                <p className="text-muted-foreground text-xs mt-3 pt-3 border-t border-border">
                  Retrieval: {[similarData.retrieval_provenance.model_name, similarData.retrieval_provenance.embedding_dim != null && `dim ${similarData.retrieval_provenance.embedding_dim}`, similarData.retrieval_provenance.timestamp && new Date(similarData.retrieval_provenance.timestamp).toLocaleString()].filter(Boolean).join(" · ")}
                </p>
              )}
            </>
          ) : (
            <p className="text-muted-foreground text-sm">No similar past incidents.</p>
          )}
        </CardContent>
      </Card>

      {canTriggerOutreach && (
        <Card className="rounded-2xl shadow-sm border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Bell className="h-4 w-4" />
              Notify caregiver
            </CardTitle>
            <p className="text-muted-foreground text-sm">
              Send a brief alert to the household caregiver (SMS/email). Preview first, then confirm.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {!outreachPreview && !outreachConfirm && (
              <Button
                className="rounded-xl"
                variant="outline"
                disabled={outreachMutation.isPending}
                onClick={() => {
                  outreachMutation.mutate(
                    { risk_signal_id: id, dry_run: true },
                    {
                      onSuccess: (res) => {
                        if (res.preview?.step_trace) {
                          const payload = (res.preview as { step_trace?: Array<{ notes?: string }> }).step_trace?.find(
                            (s) => s.notes && (s.notes.includes("channel") || s.notes.includes("consent"))
                          );
                          setOutreachPreview({
                            caregiver_message: payload?.notes ?? "Preview available.",
                            elder_safe_message: "We've shared a brief summary with your caregiver.",
                          });
                        } else {
                          setOutreachPreview({
                            caregiver_message: "Preview: caregiver will see alert summary and link to dashboard.",
                            elder_safe_message: "We've shared a brief summary with your caregiver.",
                          });
                        }
                      },
                    }
                  );
                }}
              >
                {outreachMutation.isPending ? "Loading…" : "Preview message"}
              </Button>
            )}
            {outreachPreview && !outreachConfirm && (
              <>
                <p className="text-sm text-muted-foreground">
                  Caregiver message (summary): {outreachPreview.caregiver_message}
                </p>
                <p className="text-sm text-muted-foreground">Elder-safe: {outreachPreview.elder_safe_message}</p>
                <div className="flex gap-2">
                  <Button
                    className="rounded-xl"
                    disabled={outreachMutation.isPending}
                    onClick={() => {
                      setOutreachConfirm(true);
                      outreachMutation.mutate(
                        { risk_signal_id: id, dry_run: false },
                        {
                          onSuccess: () => {
                            setOutreachPreview(null);
                            setOutreachConfirm(false);
                          },
                        }
                      );
                    }}
                  >
                    {outreachMutation.isPending ? "Sending…" : "Confirm send"}
                  </Button>
                  <Button variant="ghost" className="rounded-xl" onClick={() => setOutreachPreview(null)}>
                    Cancel
                  </Button>
                </div>
              </>
            )}
            {outreachForThisSignal.length > 0 && (
              <div className="pt-2 border-t border-border">
                <p className="text-xs font-medium text-muted-foreground mb-1">Delivery status</p>
                <ul className="space-y-1 text-sm">
                  {outreachForThisSignal.map((a: Record<string, unknown>, i: number) => (
                    <li key={i}>
                      <span className="font-medium">{(a.status as string) ?? "—"}</span>
                      {a.created_at ? ` · Created ${new Date(String(a.created_at)).toLocaleString()}` : ""}
                      {a.sent_at ? ` · Sent ${new Date(String(a.sent_at)).toLocaleString()}` : ""}
                      {a.error ? (
                        <span className="block text-destructive text-xs mt-0.5">Error: {String(a.error)}</span>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Action Plan: playbook + tasks (incident response) */}
      <Card className="rounded-2xl shadow-sm border-primary/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <ShieldAlert className="h-4 w-4" />
            Action Plan
          </CardTitle>
          <p className="text-muted-foreground text-sm">
            Guided playbook: bank contact script, device high-risk mode, tasks. Run Incident Response to create.
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          {!hasLockCardCap && capabilities && (
            <p className="text-xs text-amber-700 dark:text-amber-400 rounded-lg bg-amber-500/10 px-3 py-2">
              Your bank integration does not support lock card; we prepared the call script instead.
            </p>
          )}
          {playbookLoading && <Skeleton className="h-20 w-full rounded-xl" />}
          {!playbookLoading && !playbook && canTriggerOutreach && (
            <Button
              className="rounded-xl"
              disabled={incidentResponseMutation.isPending}
              onClick={() =>
                incidentResponseMutation.mutate({ risk_signal_id: id, dry_run: false })
              }
            >
              {incidentResponseMutation.isPending ? "Running…" : "Run Incident Response"}
            </Button>
          )}
          {!playbookLoading && playbook && (
            <div className="space-y-3">
              <p className="text-xs text-muted-foreground">
                Playbook {playbook.id?.slice(0, 8)}… · {playbook.tasks?.length ?? 0} tasks
              </p>
              <ul className="space-y-2">
                {(playbook.tasks ?? []).map((task: { id: string; task_type: string; status: string; details: Record<string, unknown> }) => (
                  <li
                    key={task.id}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-border px-3 py-2 text-sm"
                  >
                    <span className="font-medium capitalize">{task.task_type.replace(/_/g, " ")}</span>
                    <Badge variant={task.status === "done" ? "default" : "secondary"} className="rounded-lg">
                      {task.status}
                    </Badge>
                    {task.task_type === "call_bank" && task.details?.call_script && task.status !== "done" && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="rounded-lg"
                        onClick={() => {
                          navigator.clipboard.writeText(String(task.details.call_script));
                          setCopiedScript(true);
                          setTimeout(() => setCopiedScript(false), 2000);
                        }}
                      >
                        <ClipboardCopy className="h-4 w-4 mr-1" />
                        {copiedScript ? "Copied" : "Copy bank call script"}
                      </Button>
                    )}
                    {task.status !== "done" && task.status !== "blocked" && (
                      <Button
                        size="sm"
                        variant="outline"
                        className="rounded-lg"
                        disabled={completeTaskMutation.isPending}
                        onClick={() => completeTaskMutation.mutate(task.id)}
                      >
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Mark as done
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
              {detail.recommended_action && (detail.recommended_action as Record<string, unknown>).incident_packet_id && (
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <FileJson className="h-3 w-3" />
                  Bank-ready case file available (export via API /incident_packets/{(detail.recommended_action as Record<string, unknown>).incident_packet_id}).
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <AgentTrace steps={traceSteps} />

      <PolicyGateCard
        withheld={["Raw utterance text withheld until consent."]}
        couldShare={["Full transcript for caregiver review."]}
      />

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Feedback</CardTitle>
          <p className="text-muted-foreground text-sm">
            Confirm scam, false alarm, or unsure. Notes are stored for calibration.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Optional notes…"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            className="rounded-xl min-h-[80px]"
          />
          <div className="flex flex-wrap gap-2">
            <Button
              variant="destructive"
              className="rounded-xl"
              disabled={submitFeedback.isPending}
              onClick={() =>
                submitFeedback.mutate(
                  { label: "true_positive", notes: notes || undefined },
                  { onSuccess: () => setNotes("") }
                )
              }
            >
              Confirm scam
            </Button>
            <Button
              variant="outline"
              className="rounded-xl"
              disabled={submitFeedback.isPending}
              onClick={() =>
                submitFeedback.mutate(
                  { label: "false_positive", notes: notes || undefined },
                  { onSuccess: () => setNotes("") }
                )
              }
            >
              False alarm
            </Button>
            <Button
              variant="secondary"
              className="rounded-xl"
              disabled={submitFeedback.isPending}
              onClick={() =>
                submitFeedback.mutate(
                  { label: "unsure", notes: notes || undefined },
                  { onSuccess: () => setNotes("") }
                )
              }
            >
              Unsure
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
