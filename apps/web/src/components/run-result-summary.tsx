"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ExplainableIds } from "@/components/explainable-ids";
import { AlertTriangle, List, Send, ExternalLink } from "lucide-react";

export type RunResultSummaryProps = {
  /** Response from POST /investigation/run or /alerts/{id}/refresh */
  bundle: {
    created_signal_ids?: string[];
    updated_signal_ids?: string[];
    created_watchlist_ids?: string[];
    outreach_candidates?: Array<{ risk_signal_id?: string; severity?: number; decision_rule_used?: string }>;
    summary_json?: { counts?: Record<string, number>; warnings?: string[]; message?: string };
    step_trace?: Array<{ step?: string; status?: string; error?: string }>;
  };
  /** From agent_runs or mutation response */
  meta?: {
    run_id?: string | null;
    timestamp?: string | null;
    status?: string | null;
  };
  /** Callback when user clicks "Review outreach candidates" — e.g. switch to Actions tab */
  onReviewOutreach?: () => void;
  onReviewWatchlists?: () => void;
  /** Tab value for Actions (to switch to) */
  actionsTabValue?: string;
  /** When false (Family mode): outcome-only summary, no raw IDs in "What changed". When true (Explain): show technical identifiers. */
  showTechnicalIds?: boolean;
};

const statusPillVariant = (status: string | null | undefined): "default" | "secondary" | "destructive" => {
  const s = (status ?? "").toLowerCase();
  if (s === "completed" || s === "success" || s === "ok") return "default";
  if (s === "partial" || s === "warning") return "secondary";
  return "destructive";
};

export function RunResultSummary({
  bundle,
  meta,
  onReviewOutreach,
  onReviewWatchlists,
  showTechnicalIds = false,
}: RunResultSummaryProps) {
  const createdIds = bundle.created_signal_ids ?? [];
  const updatedIds = bundle.updated_signal_ids ?? [];
  const counts = bundle.summary_json?.counts ?? {};
  const newCount = (createdIds.length || counts.new_signals) ?? 0;
  const updatedCount = (updatedIds.length || counts.updated_signals) ?? 0;
  const watchlistsCount = ((bundle.created_watchlist_ids?.length ?? 0) || counts.watchlists) ?? 0;
  const outreachCount = ((bundle.outreach_candidates?.length ?? 0) || counts.outreach_candidates) ?? 0;
  const warnings = bundle.summary_json?.warnings ?? [];
  const stepTrace = bundle.step_trace ?? [];
  const hasFailedStep = stepTrace.some((s) => (s.status ?? "").toLowerCase() === "error" || (s.status ?? "").toLowerCase() === "fail");
  const statusLabel = hasFailedStep ? "Partial" : meta?.status === "completed" || !meta?.status ? "Completed" : meta?.status ?? "Completed";
  const statusPill = statusPillVariant(hasFailedStep ? "partial" : meta?.status);

  return (
    <Card className="rounded-2xl shadow-sm border-muted/50">
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="text-base">What changed</CardTitle>
          <Badge variant={statusPill} className="rounded-lg">
            {statusLabel}
          </Badge>
          {meta?.timestamp && (
            <span className="text-muted-foreground text-sm">
              {new Date(meta.timestamp).toLocaleString()}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Family-friendly outcome summary: no raw IDs. When enqueued, show message instead. */}
        <div className="rounded-xl bg-muted/40 px-4 py-3 text-sm">
          <p className="font-medium">
            {bundle.summary_json?.message ?? (
              <>
                Created {newCount} alert{newCount !== 1 ? "s" : ""} · Updated {updatedCount} alert{updatedCount !== 1 ? "s" : ""} · Added {watchlistsCount} watch item{watchlistsCount !== 1 ? "s" : ""} · Prepared {outreachCount} message{outreachCount !== 1 ? "s" : ""} (not sent).
              </>
            )}
          </p>
        </div>

        {warnings.length > 0 && (
          <div className="flex items-start gap-2 text-amber-700 dark:text-amber-400 text-sm">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <ul className="list-disc list-inside">
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {hasFailedStep && (
          <div className="text-sm text-muted-foreground">
            <p className="font-medium mb-1">Stages with issues:</p>
            <ul className="list-disc list-inside">
              {stepTrace.filter((s) => (s.status ?? "").toLowerCase() === "error" || (s.status ?? "").toLowerCase() === "fail").map((s, i) => (
                <li key={i}>{s.step ?? "step"}: {s.error ?? s.status}</li>
              ))}
            </ul>
          </div>
        )}

        {/* What changed: human summary first (New alert / Updated alert with link). Technical IDs only in Explain mode. */}
        <div>
          <p className="text-sm font-medium text-muted-foreground mb-2">Alerts and watch items</p>
          <div className="space-y-2">
            {createdIds.slice(0, 15).map((id) => (
              <AlertRow key={id} signalId={id} label="New alert" showTechnicalId={showTechnicalIds} />
            ))}
            {updatedIds.filter((id) => !createdIds.includes(id)).slice(0, 15).map((id) => (
              <AlertRow key={id} signalId={id} label="Updated alert" showTechnicalId={showTechnicalIds} />
            ))}
            {createdIds.length === 0 && updatedIds.length === 0 && newCount === 0 && updatedCount === 0 && (
              <p className="text-sm text-muted-foreground">No new or updated alerts in this run.</p>
            )}
          </div>
          {!showTechnicalIds && (createdIds.length > 0 || updatedIds.length > 0) && (
            <details className="mt-2">
              <summary className="text-xs text-muted-foreground cursor-pointer hover:underline">Show technical identifiers</summary>
              <div className="mt-2 space-y-1 font-mono text-xs text-muted-foreground">
                {createdIds.slice(0, 15).map((id) => (
                  <div key={id}>New: {id}</div>
                ))}
                {updatedIds.filter((id) => !createdIds.includes(id)).slice(0, 15).map((id) => (
                  <div key={id}>Updated: {id}</div>
                ))}
              </div>
            </details>
          )}
          {(createdIds.length > 0 || updatedIds.length > 0) && (
            <ExplainableIds
              context="alert_ids"
              items={[...createdIds, ...updatedIds.filter((id) => !createdIds.includes(id))].slice(0, 15).map((id) => ({ id, label: createdIds.includes(id) ? "New alert" : "Updated alert" }))}
              title="What these alerts are"
              className="pt-2 border-t border-border"
            />
          )}
        </div>

        {/* Next actions */}
        <div className="flex flex-wrap gap-2 pt-2 border-t border-border">
          {outreachCount > 0 && (
            <Button size="sm" className="rounded-lg" onClick={onReviewOutreach}>
              <Send className="h-4 w-4 mr-2" />
              Review outreach candidates
            </Button>
          )}
          {watchlistsCount > 0 && (
            <Button size="sm" variant="outline" className="rounded-lg" onClick={onReviewWatchlists} asChild>
              <Link href="/watchlists">
                <List className="h-4 w-4 mr-2" />
                Review watchlists
              </Link>
            </Button>
          )}
          {newCount === 0 && createdIds.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No new alerts in this window. Try adjusting the time window or run again later.
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function AlertRow({
  signalId,
  label,
  showTechnicalId,
}: {
  signalId: string;
  label: string;
  showTechnicalId?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-border bg-background/50 px-3 py-2 text-sm">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <Link href={`/alerts/${signalId}`} className="font-medium text-primary hover:underline truncate min-w-0">
        {showTechnicalId ? `${signalId.slice(0, 8)}…` : "Open alert"}
      </Link>
      <Link href={`/alerts/${signalId}`}>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0 shrink-0">
          <ExternalLink className="h-3.5 w-3.5" />
        </Button>
      </Link>
    </div>
  );
}
