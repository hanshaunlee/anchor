"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import { useSessions } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { SessionsDataSummary } from "@/components/sessions-data-summary";
import { SessionTimelineItem } from "@/components/session-timeline-item";
import { SessionsMetadataPanel } from "@/components/sessions-metadata-panel";
import type { SessionListItem } from "@/lib/api/schemas";

const PRESETS = [
  { label: "Today", days: 0 },
  { label: "Past 3 days", days: 3 },
  { label: "Past 7 days", days: 7 },
  { label: "Past 2 weeks", days: 14 },
  { label: "Past 30 days", days: 30 },
] as const;

function toDateInputValue(d: Date) {
  return d.toISOString().slice(0, 10);
}

export default function SessionsPage() {
  const demoMode = useAppStore((s) => s.demoMode);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [selectedSession, setSelectedSession] = useState<SessionListItem | null>(null);
  const [visibleCount, setVisibleCount] = useState(10);
  const VISIBLE_INCREMENT = 10;

  const setPreset = useCallback((days: number) => {
    const end = new Date();
    const start = new Date();
    if (days === 0) {
      start.setHours(0, 0, 0, 0);
    } else {
      start.setDate(start.getDate() - days);
      start.setHours(0, 0, 0, 0);
    }
    setFrom(toDateInputValue(start));
    setTo(toDateInputValue(end));
  }, []);

  const { data, isLoading } = useSessions({
    from: from || undefined,
    to: to || undefined,
    limit: 100,
  });

  const sessions = useMemo(() => {
    const list = data?.sessions ?? [];
    return [...list].sort(
      (a, b) => new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
    );
  }, [data?.sessions]);

  const visibleSessions = sessions.slice(0, visibleCount);
  const hasMoreSessions = visibleCount < sessions.length;
  const showMore = () => setVisibleCount((n) => n + VISIBLE_INCREMENT);

  useEffect(() => {
    setVisibleCount(10);
  }, [from, to]);

  return (
    <div
      className="relative space-y-6 rounded-2xl min-h-[60vh]"
      style={{
        backgroundImage: "radial-gradient(circle at 1px 1px, hsl(var(--muted) / 0.25) 0.5px, transparent 0)",
        backgroundSize: "20px 20px",
      }}
    >
      <div className="relative space-y-6">
      {demoMode && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-800 dark:text-amber-200">
          <span className="font-medium">Demo Data Mode</span> – Using fixture data. Layout matches live mode.
        </div>
      )}

      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Data capture timeline</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Chronological evidence intake. Select a session to inspect metadata and open the forensic trace.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-2">
          <Label className="text-xs font-medium text-muted-foreground">From</Label>
          <Input
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            className="rounded-xl w-[160px] font-mono"
          />
        </div>
        <div className="space-y-2">
          <Label className="text-xs font-medium text-muted-foreground">To</Label>
          <Input
            type="date"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            className="rounded-xl w-[160px] font-mono"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="text-xs font-medium text-muted-foreground self-center pb-2 hidden sm:inline">Quick:</span>
          {PRESETS.map(({ label, days }) => (
            <Button
              key={label}
              variant="outline"
              size="sm"
              className="rounded-xl font-mono text-xs"
              onClick={() => setPreset(days)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>

      <SessionsDataSummary />

      <div
        className="grid gap-6"
        style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 30%)" }}
      >
        <div className="min-w-0 rounded-2xl border border-border bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border bg-muted/20">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Chronological session timeline
            </h2>
          </div>
          <div className="p-4">
            {isLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <Skeleton key={i} className="h-24 w-full rounded-xl" />
                ))}
              </div>
            ) : sessions.length === 0 ? (
              <p className="text-muted-foreground py-12 text-center text-sm">
                No sessions in range.
              </p>
            ) : (
              <>
                <ul className="space-y-2">
                  {visibleSessions.map((s) => (
                    <li key={s.id}>
                      <SessionTimelineItem
                        session={s}
                        selected={selectedSession?.id === s.id}
                        onSelect={() => setSelectedSession(s)}
                      />
                    </li>
                  ))}
                </ul>
                {hasMoreSessions && (
                  <Button
                    variant="outline"
                    className="mt-4 w-full rounded-xl"
                    onClick={showMore}
                  >
                    Show more ({sessions.length - visibleCount} remaining)
                  </Button>
                )}
              </>
            )}
          </div>
        </div>

        <div className="min-w-0 flex flex-col">
          <SessionsMetadataPanel session={selectedSession} />
        </div>
      </div>
      </div>
    </div>
  );
}
