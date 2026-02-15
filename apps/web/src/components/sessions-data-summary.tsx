"use client";

import { useMemo } from "react";
import { useSessions, useRiskSignals, useHouseholdConsent, useAgentsStatus } from "@/hooks/use-api";
import { cn } from "@/lib/utils";
import { Activity, AlertTriangle, Clock } from "lucide-react";

const now = new Date();
const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
const from7d = sevenDaysAgo.toISOString().slice(0, 10);
const toToday = now.toISOString().slice(0, 10);

export function SessionsDataSummary({ className }: { className?: string }) {
  const { data: sessionsData } = useSessions({ from: from7d, to: toToday, limit: 500 });
  const { data: signalsData } = useRiskSignals({ limit: 1 });
  const { data: consent } = useHouseholdConsent();
  const { data: agentsData } = useAgentsStatus();

  const stats = useMemo(() => {
    const sessions = sessionsData?.sessions ?? [];
    const total = sessions.length;
    const online = sessions.filter((s) => s.mode === "online").length;
    const offline = total - online;
    const withConsent = sessions.filter((s) => s.consent_state && Object.keys(s.consent_state).length > 0).length;
    const redactedPct = total > 0 ? Math.round((withConsent / total) * 100) : 0;
    const outboundAllowed = consent?.allow_outbound_contact ?? false;
    const riskSignalsDerived = signalsData?.total ?? 0;
    const supervisor = agentsData?.agents?.find((a: { agent_name: string }) => a.agent_name === "supervisor");
    const lastRun = supervisor?.last_run_at ?? null;

    return {
      total,
      onlinePct: total > 0 ? Math.round((online / total) * 100) : 0,
      offlinePct: total > 0 ? Math.round((offline / total) * 100) : 0,
      redactedPct,
      outboundAllowed,
      riskSignalsDerived,
      lastRun,
    };
  }, [sessionsData, signalsData, consent, agentsData]);

  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-muted/20 px-4 py-3 flex flex-wrap items-center gap-x-8 gap-y-2",
        className
      )}
    >
      <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
        <Activity className="h-4 w-4" />
        Data health summary
      </div>
      <div className="flex flex-wrap items-center gap-6 text-xs">
        <span className="tabular-nums">
          <span className="text-muted-foreground">Sessions (7d):</span>{" "}
          <span className="font-medium text-foreground">{stats.total}</span>
        </span>
        <span className="tabular-nums">
          <span className="text-muted-foreground">Online:</span>{" "}
          <span className="font-medium text-emerald-600 dark:text-emerald-400">{stats.onlinePct}%</span>
          <span className="text-muted-foreground ml-0.5">/ Offline:</span>{" "}
          <span className="font-medium text-foreground">{stats.offlinePct}%</span>
        </span>
        <span className="tabular-nums">
          <span className="text-muted-foreground">Consent set:</span>{" "}
          <span className="font-medium text-foreground">{stats.redactedPct}%</span>
        </span>
        <span>
          <span className="text-muted-foreground">Outbound:</span>{" "}
          {stats.outboundAllowed ? (
            <span className="font-medium text-emerald-600 dark:text-emerald-400">Allowed</span>
          ) : (
            <span className="font-medium text-muted-foreground">Off</span>
          )}
        </span>
        <span className="tabular-nums flex items-center gap-1">
          <AlertTriangle className="h-3.5 w-3 text-muted-foreground" />
          <span className="text-muted-foreground">Risk signals derived:</span>{" "}
          <span className="font-medium text-foreground">{stats.riskSignalsDerived}</span>
        </span>
        {stats.lastRun && (
          <span className="tabular-nums flex items-center gap-1 text-muted-foreground">
            <Clock className="h-3.5 w-3" />
            Last run: {new Date(stats.lastRun).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" })}
          </span>
        )}
      </div>
    </div>
  );
}
