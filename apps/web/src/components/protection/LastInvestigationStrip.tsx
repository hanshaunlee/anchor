"use client";

import Link from "next/link";
import { useAgentsStatus } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { ExternalLink } from "lucide-react";

const SUPERVISOR = "supervisor";

export function LastInvestigationStrip() {
  const { data, isLoading } = useAgentsStatus();
  const agent = data?.agents?.find((a: { agent_name: string }) => a.agent_name === SUPERVISOR);
  const summary = agent?.last_run_summary as {
    summary_json?: { counts?: { new_signals?: number; watchlists?: number; rings?: number } };
    created_signal_ids?: string[];
  } | null;
  const counts = summary?.summary_json?.counts ?? {};
  const alerts = counts.new_signals ?? summary?.created_signal_ids?.length ?? 0;
  const watchlistChanges = counts.watchlists ?? 0;
  const ringUpdates = counts.rings ?? 0;
  const runId = agent?.last_run_id;
  const completedAt = agent?.last_run_at;
  const status = agent?.last_run_status;

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-2 border-b border-border">
        <Skeleton className="h-5 w-48" />
      </div>
    );
  }
  if (!completedAt || status !== "success") {
    return null;
  }

  const timeStr = formatTime(completedAt);
  const parts: string[] = [];
  if (alerts > 0) parts.push(`${alerts} alert${alerts !== 1 ? "s" : ""} created`);
  if (watchlistChanges > 0) parts.push(`${watchlistChanges} watchlist change${watchlistChanges !== 1 ? "s" : ""}`);
  if (ringUpdates > 0) parts.push(`${ringUpdates} ring${ringUpdates !== 1 ? "s" : ""} updated`);
  const summaryLine = parts.length ? parts.join(" · ") : "completed";

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2 border-b border-border text-sm text-muted-foreground">
      <span>
        Last Investigation: completed {timeStr}
        {summaryLine !== "completed" && ` · ${summaryLine}`}
      </span>
      {runId && (
        <Link
          href={`/agents?trace=${runId}`}
          className="inline-flex items-center gap-1 text-primary hover:underline"
        >
          View trace
          <ExternalLink className="h-3 w-3" />
        </Link>
      )}
    </div>
  );
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}
