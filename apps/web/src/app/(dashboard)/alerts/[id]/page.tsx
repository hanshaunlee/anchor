"use client";

import React, { use } from "react";
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
} from "@/hooks/use-api";
import { GraphEvidence } from "@/components/graph-evidence";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
import { PolicyGateCard } from "@/components/policy-gate-card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { ArrowLeft } from "lucide-react";

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

export default function AlertDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data: detail, isLoading } = useRiskSignalDetail(id);
  const sessionId = detail?.session_ids?.[0];
  const { data: eventsData } = useSessionEvents(sessionId ?? null, { limit: 10 });
  const { data: similarData } = useSimilarIncidents(id, 5);
  const submitFeedback = useSubmitFeedback(id);
  const [notes, setNotes] = React.useState("");
  const events = eventsData?.events ?? [];

  if (isLoading || !detail) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  const expl = detail.explanation as Record<string, unknown>;
  const motifs = (expl.motif_tags as string[]) ?? (expl.motifs as string[]) ?? [];
  const recommended = (detail.recommended_action as Record<string, unknown>) ?? {};
  const steps = (recommended.steps as string[]) ?? [];
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
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{(expl.summary as string) ?? "—"}</p>
            <p className="text-muted-foreground text-sm mt-2">
              Score: {(detail.score * 100).toFixed(0)}%
            </p>
          </CardContent>
        </Card>
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
        <GraphEvidence
          nodes={detail.subgraph.nodes}
          edges={detail.subgraph.edges}
          onPlayPath={() => {}}
        />
      )}

      {similarData?.similar && similarData.similar.length > 0 && (
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Similar incidents</CardTitle>
            <p className="text-muted-foreground text-sm">Past signals with similar patterns</p>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {similarData.similar.map((s) => (
                <li key={s.risk_signal_id} className="flex justify-between rounded-lg border border-border px-3 py-2 text-sm">
                  <Link href={`/alerts/${s.risk_signal_id}`} className="hover:underline">
                    {String(s.risk_signal_id).slice(0, 8)}…
                  </Link>
                  <span className="text-muted-foreground">
                    similarity {(s.score * 100).toFixed(0)}% · {s.outcome ?? "open"}
                  </span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

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
