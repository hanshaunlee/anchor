"use client";

import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RiskMeter } from "@/components/ui/risk-meter";
import { cn } from "@/lib/utils";
import { ArrowLeft, RefreshCw, Bell, Network } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

const statusVariant: Record<string, string> = {
  open: "bg-status-open/15 text-status-open border-status-open/30",
  acknowledged: "bg-status-acknowledged/15 text-status-acknowledged border-status-acknowledged/30",
  dismissed: "bg-status-dismissed/15 text-status-dismissed border-status-dismissed/30",
  escalated: "bg-status-escalated/15 text-status-escalated border-status-escalated/30",
};

const severityLabel: Record<number, string> = {
  1: "Low",
  2: "Low–Medium",
  3: "Medium",
  4: "High",
  5: "Critical",
};

/** Human-readable incident title from signal_type */
function incidentTitle(signalType: string): string {
  const map: Record<string, string> = {
    ring_candidate: "Borrowing Ring Alert",
    drift_warning: "Drift Warning",
    watchlist_embedding_match: "Watchlist Match",
  };
  return map[signalType] ?? signalType.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export type IncidentHeaderProps = {
  signalId: string;
  signalType: string;
  status: string;
  severity: number;
  score: number;
  ts: string;
  /** One-line narrative or summary for the bar */
  summaryLine?: string | null;
  /** Decision rule / model caption for meter */
  decisionRuleUsed?: string | null;
  /** Allow refresh investigation */
  canRefreshInvestigation: boolean;
  /** Allow notify caregiver */
  canNotifyCaregiver: boolean;
  demoMode: boolean;
  onRefreshInvestigation: () => void;
  onNotifyCaregiver: () => void;
  refreshPending: boolean;
  /** Ring ID if signal_type is ring_candidate */
  ringId?: string | null;
  /** Left border severity accent */
  severityAccent?: boolean;
  /** Judge mode: show "Show Internals" toggle */
  judgeMode?: boolean;
  onJudgeModeChange?: (v: boolean) => void;
};

export function IncidentHeader({
  signalId,
  signalType,
  status,
  severity,
  score,
  ts,
  summaryLine,
  decisionRuleUsed,
  canRefreshInvestigation,
  canNotifyCaregiver,
  demoMode,
  onRefreshInvestigation,
  onNotifyCaregiver,
  refreshPending,
  ringId,
  severityAccent = true,
  judgeMode,
  onJudgeModeChange,
}: IncidentHeaderProps) {
  const title = incidentTitle(signalType);
  const lowConfidence = score < 0.2;

  return (
    <header
      className={cn(
        "rounded-2xl border border-border bg-card shadow-sm overflow-hidden",
        severityAccent && "border-l-4",
        severity === 1 && "border-l-severity-1",
        severity === 2 && "border-l-severity-2",
        severity === 3 && "border-l-severity-3",
        severity === 4 && "border-l-severity-4",
        severity === 5 && "border-l-severity-5"
      )}
    >
      <div className="p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Link
                href="/alerts"
                className="rounded-xl p-2 hover:bg-accent inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
              >
                <ArrowLeft className="h-4 w-4" />
                Back
              </Link>
              <h1 className="text-xl font-semibold tracking-tight truncate">
                {title}
              </h1>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge className={cn("rounded-lg border", statusVariant[status] ?? "bg-muted")}>
                {status}
              </Badge>
              <span className="rounded-lg px-2 py-0.5 text-xs font-medium bg-muted text-muted-foreground">
                Severity: {severityLabel[severity] ?? severity}
              </span>
              <span className="rounded-lg px-2 py-0.5 text-xs font-medium bg-muted text-muted-foreground">
                Source: {signalType.replace(/_/g, " ")}
              </span>
              {ringId && (
                <Link href={`/rings/${ringId}`}>
                  <span className="rounded-lg bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-700 dark:text-amber-400 border border-amber-500/30 hover:bg-amber-500/25 cursor-pointer inline-block">
                    View pattern
                  </span>
                </Link>
              )}
            </div>
            {summaryLine && (
              <p className="text-sm text-muted-foreground max-w-2xl">
                {summaryLine}
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              {new Date(ts).toLocaleString()}
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:items-end shrink-0">
            <div className="w-full sm:w-40">
              <RiskMeter
                score={score}
                severity={severity}
                decisionRuleUsed={decisionRuleUsed}
                caption={lowConfidence ? "Low confidence — structural anomaly only" : undefined}
              />
            </div>
            <div className="flex flex-wrap items-center gap-3">
              {onJudgeModeChange != null && (
                <div className="flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 bg-muted/30">
                  <Switch
                    id="judge-mode"
                    checked={judgeMode ?? false}
                    onCheckedChange={onJudgeModeChange}
                  />
                  <Label htmlFor="judge-mode" className="text-xs font-medium cursor-pointer">
                    Show internals
                  </Label>
                </div>
              )}
              {canRefreshInvestigation && !demoMode && (
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-xl"
                  disabled={refreshPending}
                  onClick={onRefreshInvestigation}
                >
                  <RefreshCw className={cn("h-4 w-4 mr-2", refreshPending && "animate-spin")} />
                  Run Investigation Again
                </Button>
              )}
              {canNotifyCaregiver && !demoMode && (
                <Button size="sm" className="rounded-xl" onClick={onNotifyCaregiver}>
                  <Bell className="h-4 w-4 mr-2" />
                  Notify Caregiver
                </Button>
              )}
              <Link href={ringId ? `/graph?highlightRing=${encodeURIComponent(ringId)}` : "/graph"}>
                <Button variant="outline" size="sm" className="rounded-xl">
                  <Network className="h-4 w-4 mr-2" />
                  {ringId ? "Open in Graph" : "View in Graph"}
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
