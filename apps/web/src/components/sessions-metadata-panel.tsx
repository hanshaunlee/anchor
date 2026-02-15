"use client";

import { useSessionEvents } from "@/hooks/use-api";
import type { SessionListItem } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";
import { FileText, Clock, Hash, Layers } from "lucide-react";

export type SessionsMetadataPanelProps = {
  session: SessionListItem | null;
  /** When we have risk signals linked to this session (optional). */
  riskSignalsCount?: number;
  className?: string;
};

export function SessionsMetadataPanel({
  session,
  riskSignalsCount,
  className,
}: SessionsMetadataPanelProps) {
  const { data: eventsData } = useSessionEvents(session?.id ?? null, { limit: 500 });
  const events = eventsData?.events ?? [];
  const total = eventsData?.total ?? 0;
  const hasConsent = session?.consent_state && Object.keys(session.consent_state).length > 0;

  if (!session) {
    return (
      <div
        className={cn(
          "rounded-xl border border-border bg-muted/10 px-4 py-6 text-center text-sm text-muted-foreground",
          className
        )}
      >
        <p>Hover or select a session to see metadata.</p>
      </div>
    );
  }

  const started = new Date(session.started_at);
  const ended = session.ended_at ? new Date(session.ended_at) : null;
  const durationMs = ended ? ended.getTime() - started.getTime() : null;
  const durationStr =
    durationMs != null
      ? durationMs >= 60_000
        ? `${Math.floor(durationMs / 60_000)}m ${Math.round((durationMs % 60_000) / 1000)}s`
        : `${Math.round(durationMs / 1000)}s`
      : "—";

  return (
    <div className={cn("rounded-xl border border-border bg-card overflow-hidden", className)}>
      <div className="px-4 py-3 border-b border-border bg-muted/20">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Session metrics
        </h3>
      </div>
      <div className="p-4 space-y-4 text-sm">
        <div className="flex items-start gap-2">
          <Hash className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5" />
          <div className="min-w-0">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Session ID</p>
            <p className="font-mono text-xs truncate" title={session.id}>
              {session.id}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 shrink-0 text-muted-foreground" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Duration</p>
            <p className="font-mono text-xs">{durationStr}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 shrink-0 text-muted-foreground" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Event count</p>
            <p className="font-mono text-xs">{total}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Consent state</p>
            <p className="text-xs">{hasConsent ? "Set (redaction applied)" : "Default"}</p>
          </div>
        </div>
        {riskSignalsCount != null && riskSignalsCount > 0 && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Risk signals</p>
            <p className="font-mono text-xs">{riskSignalsCount} derived</p>
          </div>
        )}
      </div>
    </div>
  );
}
