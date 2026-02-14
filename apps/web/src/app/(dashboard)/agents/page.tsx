"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
import { PolicyGateCard } from "@/components/policy-gate-card";
import {
  useAgentsStatus,
  useFinancialRunMutation,
  useFinancialTrace,
} from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import { Bot, Play, RefreshCw } from "lucide-react";
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

function logsToTraceSteps(logs: string[]): TraceStep[] {
  return logs.map((line, i) => ({
    step: `Step ${i + 1}`,
    description: line,
    status: "success" as const,
  }));
}

export default function AgentsPage() {
  const [dryRunInput, setDryRunInput] = useState("");
  const [lastRunLogs, setLastRunLogs] = useState<TraceStep[]>(MOCK_TRACE);
  const [lastRunId, setLastRunId] = useState<string | null>(null);
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
  const { data: statusData, isLoading: statusLoading } = useAgentsStatus();
  const runMutation = useFinancialRunMutation();
  const { data: traceData } = useFinancialTrace(lastRunId);

  const agents = statusData?.agents ?? [];
  const traceStepsFromApi =
    traceData?.summary_json && Array.isArray((traceData.summary_json as { logs?: string[] }).logs)
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
          setLastRunId(res.run_id ?? null);
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
          setLastRunId(res.run_id ?? null);
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
                {agents.map((a) => (
                  <li key={a.agent_name} className="flex justify-between rounded-xl border border-border px-4 py-2 text-sm">
                    <span className="font-medium">{a.agent_name}</span>
                    <span className="text-muted-foreground">
                      {a.last_run_at
                        ? new Date(a.last_run_at).toLocaleString()
                        : "Never"}
                      {a.last_run_status != null ? ` · ${a.last_run_status}` : ""}
                    </span>
                  </li>
                ))}
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

      <AgentTrace steps={traceStepsFromApi} />

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
