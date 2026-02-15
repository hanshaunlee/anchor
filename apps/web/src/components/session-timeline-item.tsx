"use client";

import Link from "next/link";
import type { SessionListItem } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";
import { ChevronRight } from "lucide-react";

export type SessionTimelineItemProps = {
  session: SessionListItem;
  /** When available from parent (e.g. joined with risk signals). */
  derivedSignalCount?: number;
  derivedSeverity?: "low" | "medium" | "high";
  /** When available (e.g. from signal or backend). */
  extractedEntitiesCount?: number;
  selected?: boolean;
  onSelect?: () => void;
};

export function SessionTimelineItem({
  session,
  derivedSignalCount,
  derivedSeverity,
  extractedEntitiesCount,
  selected,
  onSelect,
}: SessionTimelineItemProps) {
  const started = new Date(session.started_at);
  const dateStr = started.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  const timeStr = started.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
  const dateTimeStr = `${dateStr} · ${timeStr}`;
  const isOnline = session.mode === "online";
  const hasConsent = session.consent_state && Object.keys(session.consent_state).length > 0;
  const consentLabel = hasConsent ? "Consent set" : "Default consent";
  const summaryLine = session.summary_text ?? "—";
  const derivedLabel =
    derivedSignalCount != null && derivedSignalCount > 0
      ? `Risk signal${derivedSignalCount > 1 ? "s" : ""} → ${derivedSeverity ?? "—"}`
      : "None";
  const entitiesLabel =
    extractedEntitiesCount != null ? `Entities: ${extractedEntitiesCount}` : "Entities: —";

  return (
    <div
      className={cn(
        "group relative flex gap-3 rounded-xl border transition-colors",
        selected ? "border-primary/50 bg-primary/5" : "border-border bg-card hover:bg-muted/30"
      )}
      onMouseEnter={onSelect}
    >
      {/* Timeline dot + line */}
      <div className="flex shrink-0 flex-col items-center pt-5">
        <div
          className={cn(
            "h-3 w-3 rounded-full border-2 border-background ring-2",
            isOnline ? "bg-emerald-500 ring-emerald-500/30" : "bg-muted-foreground/50 ring-muted/30"
          )}
        />
        <div className="mt-1 w-px flex-1 min-h-[2rem] bg-border" />
      </div>

      <div className="flex-1 min-w-0 py-3 pr-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm font-semibold text-foreground" title={started.toISOString()}>
            {dateTimeStr}
          </span>
          <span
            className={cn(
              "rounded-md px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider",
              isOnline ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400" : "bg-muted text-muted-foreground"
            )}
          >
            {isOnline ? "Online" : "Offline"}
          </span>
          <span
            className={cn(
              "rounded-md px-2 py-0.5 text-[10px] font-medium",
              hasConsent ? "bg-amber-500/15 text-amber-700 dark:text-amber-400" : "bg-muted text-muted-foreground"
            )}
          >
            {consentLabel}
          </span>
        </div>
        <p className="text-sm text-foreground mt-1 line-clamp-2">{summaryLine}</p>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-0.5 mt-1.5 text-[11px] text-muted-foreground font-mono">
          <span>Derived: {derivedLabel}</span>
          <span>{entitiesLabel}</span>
        </div>
        <Link
          href={`/sessions/${session.id}`}
          className="inline-flex items-center gap-1 mt-2 text-xs font-medium text-primary hover:underline"
        >
          View forensic trace
          <ChevronRight className="h-3.5 w-3" />
        </Link>
      </div>
    </div>
  );
}
