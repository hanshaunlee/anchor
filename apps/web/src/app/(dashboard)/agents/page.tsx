"use client";

import { useState, useCallback } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
import { PolicyGateCard } from "@/components/policy-gate-card";
import {
  useAgentsStatus,
  useFinancialRunMutation,
  useFinancialTrace,
  useAgentTrace,
  useAgentRunMutation,
  useHouseholdMe,
  useOutreachSummary,
  AGENT_SLUG_TO_NAME,
} from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import { Bot, Play, RefreshCw, Copy } from "lucide-react";
import { motion } from "framer-motion";

const PIPELINE_STEPS = [
  "Ingest",
  "GraphUpdate",
  "Score",
  "Explain",
  "ConsentGate",
  "Watchlist",
  "EscalationDraft",
  "Persist",
];

const MOCK_TRACE: TraceStep[] = [
  { step: "Ingest", description: "Load events for household in time range.", inputs: "session_id, 5 events", outputs: "ingested_events", status: "success", latency_ms: 12 },
  { step: "Normalize", description: "Build utterances, entities, mentions from events.", outputs: "3 utterances, 2 entities", status: "success", latency_ms: 45 },
  { step: "GraphUpdate", description: "Persist entities/mentions/relationships; mark graph_updated.", status: "success", latency_ms: 28 },
  { step: "Score", description: "Run GNN risk scoring; append risk_scores.", outputs: "risk_score 0.82", status: "success", latency_ms: 120 },
  { step: "Explain", description: "Generate motifs and evidence subgraph.", outputs: "motifs, subgraph", status: "success", latency_ms: 85 },
  { step: "ConsentGate", description: "Check consent_state; set consent_allows_escalation.", outputs: "allowed", status: "success", latency_ms: 2 },
  { step: "Watchlist", description: "Synthesize watchlist patterns if consent allows.", outputs: "1 watchlist", status: "success", latency_ms: 15 },
  { step: "EscalationDraft", description: "Draft caregiver notification (never sends).", outputs: "draft", status: "success", latency_ms: 8 },
  { step: "Persist", description: "Write risk_signals, watchlists to DB.", status: "success", latency_ms: 35 },
];

function CopyRetrainButton({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(() => {
    void navigator.clipboard.writeText(command).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [command]);
  return (
    <Button
      size="sm"
      variant="ghost"
      className="h-7 gap-1 text-xs text-primary"
      onClick={copy}
    >
      <Copy className="h-3 w-3" />
      {copied ? "Copied!" : "Copy retrain command"}
    </Button>
  );
}

function logsToTraceSteps(logs: string[]): TraceStep[] {
  return logs.map((line, i) => ({
    step: `Step ${i + 1}`,
    description: line,
    status: "success" as const,
  }));
}

/** Convert API step_trace (agent_runs) to friendly Agent Trace steps. */
function stepTraceToTraceSteps(stepTrace: Array<{ step?: string; status?: string; error?: string; [k: string]: unknown }>): TraceStep[] {
  if (!Array.isArray(stepTrace)) return [];
  return stepTrace.map((item) => {
    const step = (item.step ?? "step") as string;
    const desc = item.error ?? (item.status === "ok" ? "Completed" : (item.status ?? "pending"));
    const status: "success" | "warn" | "fail" =
      item.status === "ok" || item.status === "success" ? "success"
        : item.status === "error" || item.status === "fail" ? "fail"
        : "warn";
    return { step, description: String(desc), status };
  });
}

const OTHER_AGENTS: { slug: "drift" | "narrative" | "ring" | "calibration" | "redteam"; label: string }[] = [
  { slug: "drift", label: "Graph Drift" },
  { slug: "narrative", label: "Evidence Narrative" },
  { slug: "ring", label: "Ring Discovery" },
  { slug: "calibration", label: "Continual Calibration" },
  { slug: "redteam", label: "Synthetic Red-Team" },
];

function OtherAgentsRunButtons({ onRunSuccess }: { onRunSuccess: (runId: string, agentName: string) => void }) {
  const driftMut = useAgentRunMutation("drift");
  const narrativeMut = useAgentRunMutation("narrative");
  const ringMut = useAgentRunMutation("ring");
  const calibrationMut = useAgentRunMutation("calibration");
  const redteamMut = useAgentRunMutation("redteam");
  const mutMap = { drift: driftMut, narrative: narrativeMut, ring: ringMut, calibration: calibrationMut, redteam: redteamMut };
  const shortLabel: Record<string, string> = { drift: "Drift", narrative: "Narrative", ring: "Ring", calibration: "Calibration", redteam: "Red-Team" };
  const mutations = OTHER_AGENTS.map(({ slug }) => ({ slug, label: shortLabel[slug] ?? slug, mut: mutMap[slug] }));
  return (
    <div className="flex flex-wrap gap-2">
      {mutations.map(({ slug, label, mut }) => (
        <div key={slug} className="flex gap-2 items-center">
          <Button
            size="sm"
            variant="outline"
            className="rounded-xl"
            disabled={mut.isPending}
            onClick={() =>
              mut.mutate(
                { dry_run: true },
                {
                  onSuccess: (res) => {
                    if (res.run_id) onRunSuccess(res.run_id, AGENT_SLUG_TO_NAME[slug]);
                  },
                }
              )
            }
          >
            {mut.isPending ? "…" : `${label} (dry)`}
          </Button>
          <Button
            size="sm"
            className="rounded-xl"
            disabled={mut.isPending}
            onClick={() =>
              mut.mutate(
                { dry_run: false },
                {
                  onSuccess: (res) => {
                    if (res.run_id) onRunSuccess(res.run_id, AGENT_SLUG_TO_NAME[slug]);
                  },
                }
              )
            }
          >
            {mut.isPending ? "…" : `Run ${label}`}
          </Button>
        </div>
      ))}
    </div>
  );
}

export default function AgentsPage() {
  const [dryRunInput, setDryRunInput] = useState("");
  const [lastRunLogs, setLastRunLogs] = useState<TraceStep[]>(MOCK_TRACE);
  const [lastRunId, setLastRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<{ runId: string; agentName: string } | null>(null);
  const [useDemoEvents, setUseDemoEvents] = useState(false);
  const [previewResult, setPreviewResult] = useState<{
    risk_signals_count: number;
    watchlists_count: number;
    risk_signals?: unknown[];
    watchlists?: unknown[];
    motif_tags?: string[];
    timeline_snippet?: unknown[];
    input_events?: unknown[];
  } | null>(null);

  const demoMode = useAppStore((s) => s.demoMode);
  const { data: me } = useHouseholdMe();
  const canSeeOutreachSummary = me?.role === "caregiver" || me?.role === "admin";
  const { data: outreachSummary } = useOutreachSummary(canSeeOutreachSummary);
  const { data: statusData, isLoading: statusLoading } = useAgentsStatus();
  const runMutation = useFinancialRunMutation();
  const traceRunId = selectedRun?.runId ?? lastRunId;
  const traceAgentName = selectedRun?.agentName ?? (lastRunId ? "financial_security" : null);
  const { data: agentTraceData } = useAgentTrace(selectedRun?.runId ?? null, selectedRun?.agentName ?? null);
  const { data: financialTraceData } = useFinancialTrace(selectedRun ? null : lastRunId);
  const traceData = selectedRun ? agentTraceData : financialTraceData;

  const agents = statusData?.agents ?? [];
  const traceStepsFromApi =
    (traceData?.step_trace && stepTraceToTraceSteps(traceData.step_trace as Array<{ step?: string; status?: string; error?: string }>).length > 0)
      ? stepTraceToTraceSteps(traceData.step_trace as Array<{ step?: string; status?: string; error?: string }>)
      : traceData?.summary_json && Array.isArray((traceData.summary_json as { logs?: string[] }).logs)
        ? logsToTraceSteps((traceData.summary_json as { logs: string[] }).logs)
        : runMutation.data?.logs?.length
          ? logsToTraceSteps(runMutation.data.logs)
          : lastRunLogs;

  const handlePreview = () => {
    if (demoMode) {
      setLastRunLogs(MOCK_TRACE);
      setPreviewResult(null);
      return;
    }
    runMutation.mutate(
      { time_window_days: 7, dry_run: true, use_demo_events: useDemoEvents },
      {
        onSuccess: (res) => {
          setLastRunLogs(res.logs?.length ? logsToTraceSteps(res.logs) : MOCK_TRACE);
          const rid = res.run_id ?? null;
          setLastRunId(rid);
          if (rid) setSelectedRun({ runId: rid, agentName: "financial_security" });
          setPreviewResult({
            risk_signals_count: res.risk_signals_count ?? 0,
            watchlists_count: res.watchlists_count ?? 0,
            risk_signals: res.risk_signals ?? undefined,
            watchlists: res.watchlists ?? undefined,
            motif_tags: res.motif_tags ?? undefined,
            timeline_snippet: res.timeline_snippet ?? undefined,
            input_events: res.input_events ?? undefined,
          });
        },
        onError: () => {
          setLastRunLogs(MOCK_TRACE);
          setPreviewResult(null);
        },
      }
    );
  };

  const handleRun = () => {
    if (demoMode) return;
    runMutation.mutate(
      { time_window_days: 7, dry_run: false, use_demo_events: useDemoEvents },
      {
        onSuccess: (res) => {
          const rid = res.run_id ?? null;
          setLastRunId(rid);
          if (rid) setSelectedRun({ runId: rid, agentName: "financial_security" });
          setLastRunLogs(res.logs?.length ? logsToTraceSteps(res.logs) : lastRunLogs);
          setPreviewResult(null);
        },
      }
    );
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Bot className="h-7 w-7" />
          Agent Center
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          LangGraph pipeline: human-friendly view. Dry run to preview; run to persist.
        </p>
      </div>

      {!demoMode && (
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Last run status</CardTitle>
            <p className="text-muted-foreground text-sm">Agents and last run time</p>
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-16 w-full rounded-xl" />
            ) : agents.length === 0 ? (
              <p className="text-muted-foreground text-sm">No agents or not configured.</p>
            ) : (
              <ul className="space-y-2">
                {agents.map((a) => {
                  const summary = (a.last_run_summary as Record<string, unknown>) ?? {};
                  const headline = (summary.headline as string) ?? (summary.reason as string);
                  const keyMetrics = summary.key_metrics as Record<string, unknown> | undefined;
                  const keyFindings = summary.key_findings as string[] | undefined;
                  const recommendedActions = summary.recommended_actions as string[] | undefined;
                  const artifactRefs = summary.artifact_refs as Record<string, unknown[]> | undefined;
                  return (
                    <li key={a.agent_name} className="flex flex-col gap-1 rounded-xl border border-border px-4 py-2 text-sm">
                      <div className="flex justify-between">
                        <span className="font-medium">{a.agent_name.replace(/_/g, " ")}</span>
                        <span className="text-muted-foreground">
                          {a.last_run_at
                            ? new Date(a.last_run_at).toLocaleString()
                            : "Never"}
                          {a.last_run_status != null ? ` · ${a.last_run_status}` : ""}
                        </span>
                      </div>
                      {headline && <p className="text-xs font-medium text-foreground">{headline}</p>}
                      {summary.drift_detected != null && (
                        <p className="text-xs text-muted-foreground">
                          Drift: {String(summary.drift_detected)} {summary.metrics && typeof summary.metrics === "object" && "centroid_shift" in summary.metrics ? `(${(summary.metrics as { centroid_shift?: number }).centroid_shift})` : ""}
                        </p>
                      )}
                      {summary.rings_found != null && (
                        <p className="text-xs text-muted-foreground">Rings: {String(summary.rings_found)}</p>
                      )}
                      {summary.regression_pass_rate != null && (
                        <p className="text-xs text-muted-foreground">Red-team pass rate: {(Number(summary.regression_pass_rate) * 100).toFixed(0)}%</p>
                      )}
                      {summary.regression_passed === false && (summary.failing_cases as unknown[])?.length > 0 && (
                        <p className="text-xs text-destructive">{(summary.failing_cases as unknown[]).length} failing case(s)</p>
                      )}
                      {(summary.sent === true || summary.suppressed === true) && (
                        <p className="text-xs text-muted-foreground">
                          Outreach: {summary.sent ? "sent" : ""}
                          {summary.suppressed ? " suppressed" : ""}
                        </p>
                      )}
                      {keyMetrics && Object.keys(keyMetrics).length > 0 && (
                        <p className="text-xs text-muted-foreground">
                          Metrics: {Object.entries(keyMetrics).map(([k, v]) => `${k}=${v}`).join(", ")}
                        </p>
                      )}
                      {keyFindings?.length ? (
                        <ul className="text-xs text-muted-foreground list-disc list-inside">{keyFindings.slice(0, 3).map((f, i) => <li key={i}>{f}</li>)}</ul>
                      ) : null}
                      {recommendedActions?.length ? (
                        <p className="text-xs text-muted-foreground">Actions: {recommendedActions.slice(0, 2).join("; ")}</p>
                      ) : null}
                      {artifactRefs && (artifactRefs.risk_signal_ids?.length || artifactRefs.ring_ids?.length || artifactRefs.replay_fixture_path || artifactRefs.narrative_report_id) ? (
                        <p className="text-xs text-primary">
                          Artifacts:{" "}
                          {(artifactRefs.risk_signal_ids as string[])?.length ? ` ${(artifactRefs.risk_signal_ids as string[]).length} signal(s)` : ""}
                          {(artifactRefs.ring_ids as string[])?.length ? ` ${(artifactRefs.ring_ids as string[]).length} ring(s)` : ""}
                          {artifactRefs.replay_fixture_path ? " replay" : ""}
                          {artifactRefs.narrative_report_id ? " report" : ""}
                        </p>
                      ) : null}
                      {a.agent_name === "evidence_narrative" && artifactRefs?.narrative_report_id && (
                        <Link href={`/reports/narrative/${artifactRefs.narrative_report_id}`}>
                          <Button size="sm" variant="outline" className="h-7 text-xs">
                            View report
                          </Button>
                        </Link>
                      )}
                      {a.agent_name === "continual_calibration" && (summary.key_metrics != null || summary.calibration_report != null) && (
                        <Link href="/reports/calibration">
                          <Button size="sm" variant="outline" className="h-7 text-xs">
                            View calibration report
                          </Button>
                        </Link>
                      )}
                      {a.agent_name === "synthetic_redteam" && (summary.regression_pass_rate != null || summary.scenarios_generated != null) && (
                        <Link href="/reports/redteam">
                          <Button size="sm" variant="outline" className="h-7 text-xs">
                            View red-team report
                          </Button>
                        </Link>
                      )}
                      {a.agent_name === "graph_drift" && (summary.retrain_command as string) && (
                        <CopyRetrainButton command={summary.retrain_command as string} />
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </CardContent>
        </Card>
      )}

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Pipeline steps</CardTitle>
          <p className="text-muted-foreground text-sm">
            Ingest → GraphUpdate → Score → Explain → ConsentGate → Watchlist → EscalationDraft → Persist
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {PIPELINE_STEPS.map((step, i) => (
              <motion.span
                key={step}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="rounded-xl bg-muted px-3 py-2 text-sm font-medium"
              >
                {step}
              </motion.span>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="flex flex-wrap items-center gap-3">
        {!demoMode && (
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={useDemoEvents}
              onChange={(e) => setUseDemoEvents(e.target.checked)}
              className="rounded border-border"
            />
            Use demo events (synthetic scam scenario)
          </label>
        )}
        <Button
          className="rounded-xl"
          onClick={handlePreview}
          disabled={runMutation.isPending}
        >
          <Play className="h-4 w-4 mr-2" />
          {runMutation.isPending ? "Running…" : "Preview pipeline (dry run)"}
        </Button>
        {!demoMode && (
          <Button
            variant="outline"
            className="rounded-xl"
            onClick={handleRun}
            disabled={runMutation.isPending}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Run pipeline
          </Button>
        )}
      </div>

      {previewResult && (
        <Card className="rounded-2xl shadow-sm border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Dry run result</CardTitle>
            <p className="text-muted-foreground text-sm">
              {previewResult.risk_signals_count} risk signal(s), {previewResult.watchlists_count} watchlist(s).
              {previewResult.input_events != null ? ` ${previewResult.input_events.length} demo input events.` : ""}
              Not persisted.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {previewResult.motif_tags && previewResult.motif_tags.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Motif tags</p>
                <div className="flex flex-wrap gap-2">
                  {previewResult.motif_tags.map((t, i) => (
                    <span key={i} className="rounded-lg bg-muted px-2 py-1 text-xs">{t}</span>
                  ))}
                </div>
              </div>
            )}
            {previewResult.risk_signals && previewResult.risk_signals.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Risk signals</p>
                <pre className="rounded-xl bg-muted p-3 text-xs overflow-auto max-h-40">
                  {JSON.stringify(previewResult.risk_signals, null, 2)}
                </pre>
              </div>
            )}
            {previewResult.watchlists && previewResult.watchlists.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Watchlists</p>
                <pre className="rounded-xl bg-muted p-3 text-xs overflow-auto max-h-40">
                  {JSON.stringify(previewResult.watchlists, null, 2)}
                </pre>
              </div>
            )}
            {previewResult.input_events != null && previewResult.input_events.length > 0 && (
              <details className="rounded-xl bg-muted/50 p-3">
                <summary className="text-xs font-medium cursor-pointer">Input events ({previewResult.input_events.length})</summary>
                <pre className="mt-2 text-xs overflow-auto max-h-48">
                  {JSON.stringify(previewResult.input_events, null, 2)}
                </pre>
              </details>
            )}
          </CardContent>
        </Card>
      )}

      <div className="space-y-2">
        {traceRunId && traceAgentName && (
          <p className="text-muted-foreground text-sm">
            Viewing trace: <span className="font-medium">{traceAgentName.replace(/_/g, " ")}</span>
            {selectedRun && (
              <button
                type="button"
                className="ml-2 text-xs underline hover:no-underline"
                onClick={() => setSelectedRun(null)}
              >
                Show Financial only
              </button>
            )}
          </p>
        )}
        <AgentTrace steps={traceStepsFromApi} />
      </div>

      {!demoMode && canSeeOutreachSummary && outreachSummary && (
        <Card className="rounded-2xl shadow-sm border-primary/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Outreach Agent</CardTitle>
            <p className="text-muted-foreground text-sm">Sent / suppressed / failed counts and recent outbound actions</p>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap gap-4 text-sm">
              <span className="rounded-lg bg-green-500/15 px-2 py-1 text-green-700 dark:text-green-400">Sent: {outreachSummary.counts?.sent ?? 0}</span>
              <span className="rounded-lg bg-amber-500/15 px-2 py-1 text-amber-700 dark:text-amber-400">Suppressed: {outreachSummary.counts?.suppressed ?? 0}</span>
              <span className="rounded-lg bg-destructive/15 px-2 py-1 text-destructive">Failed: {outreachSummary.counts?.failed ?? 0}</span>
              <span className="rounded-lg bg-muted px-2 py-1">Queued: {outreachSummary.counts?.queued ?? 0}</span>
            </div>
            {outreachSummary.recent && outreachSummary.recent.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Recent</p>
                <ul className="space-y-1 text-xs">
                  {outreachSummary.recent.slice(0, 5).map((r: { id: string; status: string; created_at: string | null; sent_at: string | null; error: string | null; channel: string | null }) => (
                    <li key={r.id} className="flex flex-wrap gap-2">
                      <span className="font-medium">{r.status}</span>
                      {r.channel && <span>{r.channel}</span>}
                      {r.created_at && <span>{new Date(r.created_at).toLocaleString()}</span>}
                      {r.error && <span className="text-destructive">{r.error}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {!demoMode && (
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Other agents</CardTitle>
            <p className="text-muted-foreground text-sm">
              Run drift, narrative, ring discovery, calibration, or red-team. Trace appears above when a run returns a run_id.
            </p>
          </CardHeader>
          <CardContent>
            <OtherAgentsRunButtons
              onRunSuccess={(runId, agentName) => setSelectedRun({ runId, agentName })}
            />
          </CardContent>
        </Card>
      )}

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Dry run (paste input)</CardTitle>
          <p className="text-muted-foreground text-sm">
            Optional: paste a transcript or JSON event batch. Preview runs on server with last 7 days of data.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder='Optional: [{"event_type": "final_asr", "payload": {"text": "..."}}]'
            value={dryRunInput}
            onChange={(e) => setDryRunInput(e.target.value)}
            className="min-h-[100px] rounded-xl font-mono text-sm"
          />
          <p className="text-muted-foreground text-xs">
            Outputs: risk signal preview, watchlist preview, escalation draft (never sent).
          </p>
        </CardContent>
      </Card>

      <PolicyGateCard
        withheld={["Raw transcript withheld until consent."]}
        couldShare={["Full pipeline trace for caregiver."]}
      />
    </div>
  );
}
