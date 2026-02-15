"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useRiskSignals, useHouseholdMe, useSessions, useSummaries } from "@/hooks/use-api";
import { useRiskSignalStream } from "@/hooks/use-risk-signal-stream";
import { Skeleton } from "@/components/ui/skeleton";
import { useAppStore } from "@/store/use-app-store";
import {
  AlertTriangle,
  TrendingUp,
  Calendar,
  FileText,
  ChevronRight,
  CheckCircle2,
  MessageSquare,
  ShieldCheck,
} from "lucide-react";
import { RiskSignalCard } from "@/components/risk-signal-card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { useMemo } from "react";
import type { SessionListItem } from "@/lib/api/schemas";

function formatSessionDate(iso: string): string {
  const d = new Date(iso);
  const today = new Date();
  const isToday =
    d.getDate() === today.getDate() &&
    d.getMonth() === today.getMonth() &&
    d.getFullYear() === today.getFullYear();
  if (isToday) return "Today, " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const isYesterday =
    d.getDate() === yesterday.getDate() &&
    d.getMonth() === yesterday.getMonth() &&
    d.getFullYear() === yesterday.getFullYear();
  if (isYesterday) return "Yesterday, " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function SessionRow({ session }: { session: SessionListItem }) {
  return (
    <Link href={`/sessions/${session.id}`}>
      <div className="flex items-start gap-3 rounded-xl border border-border bg-card p-3 transition hover:bg-accent/50">
        <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <p className="text-xs text-muted-foreground">{formatSessionDate(session.started_at)}</p>
          <p className="mt-0.5 text-sm font-medium leading-snug line-clamp-2">
            {session.summary_text || "Session activity"}
          </p>
        </div>
        <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
      </div>
    </Link>
  );
}

export default function DashboardPage() {
  const { data: me } = useHouseholdMe();
  const explainMode = useAppStore((s) => s.explainMode);
  const showAdvanced = me?.role === "caregiver" || me?.role === "admin";
  const canRunSafetyCheck = me?.role === "caregiver" || me?.role === "admin";
  const { data: signalsData, isLoading: signalsLoading } = useRiskSignals({ limit: 10 });
  const { data: sessionsData, isLoading: sessionsLoading } = useSessions({ limit: 8 });
  const { data: summariesData, isLoading: summariesLoading } = useSummaries({ limit: 3 });
  useRiskSignalStream(!!me && me.role !== "elder");

  const signals = useMemo(() => signalsData?.signals ?? [], [signalsData?.signals]);
  const openSignals = useMemo(() => signals.filter((s) => s.status === "open"), [signals]);
  const openCount = openSignals.length;

  const sessions = useMemo(() => sessionsData?.sessions ?? [], [sessionsData?.sessions]);
  const todayStr = useMemo(() => new Date().toISOString().slice(0, 10), []);
  const sessionsToday = useMemo(
    () => sessions.filter((s) => s.started_at.startsWith(todayStr)),
    [sessions, todayStr]
  );
  const recentSessions = sessions.slice(0, 5);

  const latestSummary = useMemo(() => {
    const list = Array.isArray(summariesData) ? summariesData : [];
    const withPeriod = list
      .filter((s) => s.period_start != null && s.period_end != null)
      .sort((a, b) => new Date(b.period_end!).getTime() - new Date(a.period_end!).getTime());
    return withPeriod.length > 0 ? withPeriod[0] : null;
  }, [summariesData]);

  const severityCounts = useMemo(() => {
    const c: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    signals.forEach((s) => {
      c[s.severity] = (c[s.severity] ?? 0) + 1;
    });
    return Object.entries(c).map(([sev, count]) => ({ severity: `S${sev}`, count }));
  }, [signals]);

  const last7Days = useMemo(() => {
    const d: { day: string; score: number; count: number }[] = [];
    const safeDayStr = (ts: string): string | null => {
      const date = new Date(ts);
      if (Number.isNaN(date.getTime())) return null;
      return date.toISOString().slice(0, 10);
    };
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dayStr = date.toISOString().slice(0, 10);
      const daySignals = signals.filter((s) => safeDayStr(s.ts) === dayStr);
      const maxScore = daySignals.length ? Math.max(...daySignals.map((s) => s.score)) : 0;
      d.push({ day: dayStr.slice(5), score: maxScore, count: daySignals.length });
    }
    return d;
  }, [signals]);

  const displayName = me?.display_name || me?.name || "your loved one";

  return (
    <div className="space-y-8">
      {/* At a glance */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Today
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            {signalsLoading ? (
              "Loading…"
            ) : openCount > 0 ? (
              <>
                <span className="font-medium text-foreground">{openCount} alert{openCount !== 1 ? "s" : ""}</span> need
                your attention
              </>
            ) : (
              <>Overview for {displayName}</>
            )}
          </p>
        </div>
        <p className="text-muted-foreground text-sm">
          {new Date().toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric", year: "numeric" })}
        </p>
      </div>

      {/* Run Safety Check — family-friendly one button for caregivers/admins */}
      {canRunSafetyCheck && (
        <Card className="rounded-2xl shadow-sm border-primary/20">
          <CardContent className="pt-6 pb-6 flex flex-wrap items-center gap-4">
            <div className="flex-1 min-w-0">
              <p className="font-medium">Run a Safety Check</p>
              <p className="text-muted-foreground text-sm mt-0.5">
                Detect risks, add explanations, and prepare recommended next steps (nothing is sent until you approve).
              </p>
            </div>
            <Link href="/agents">
              <Button className="rounded-xl" size="lg">
                <ShieldCheck className="h-4 w-4 mr-2" />
                Run Safety Check
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {/* Needs your attention — only when there are open alerts */}
      {openCount > 0 && (
        <Card className="rounded-2xl border-amber-200/50 bg-amber-50/30 dark:border-amber-900/30 dark:bg-amber-950/20 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-500" />
              Needs your attention
            </CardTitle>
            <p className="text-muted-foreground text-sm">
              Review these alerts and take action when needed.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {openSignals.slice(0, 4).map((s) => (
              <RiskSignalCard key={s.id} signal={s} />
            ))}
            <Link href="/alerts" className="block">
              <Button variant="outline" className="w-full rounded-xl border-amber-300 dark:border-amber-700">
                View all alerts
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}

      {/* All clear when no open alerts */}
      {!signalsLoading && openCount === 0 && signals.length >= 0 && (
        <Card className="rounded-2xl border-emerald-200/50 bg-emerald-50/30 dark:border-emerald-900/30 dark:bg-emerald-950/20 shadow-sm">
          <CardContent className="flex items-center gap-3 py-4">
            <CheckCircle2 className="h-8 w-8 shrink-0 text-emerald-600 dark:text-emerald-500" />
            <div>
              <p className="font-medium text-emerald-800 dark:text-emerald-200">All clear for now</p>
              <p className="text-sm text-muted-foreground">
                No open alerts. You can still{" "}
                <Link href="/alerts" className="underline hover:no-underline">view past alerts</Link> or{" "}
                <Link href="/sessions" className="underline hover:no-underline">recent activity</Link>.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Two columns: Recent activity + This week summary */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Recent activity */}
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                Recent activity
              </CardTitle>
              <p className="text-muted-foreground text-sm mt-0.5">
                {sessionsToday.length > 0
                  ? `${sessionsToday.length} session${sessionsToday.length !== 1 ? "s" : ""} today`
                  : "Latest conversations and sessions"}
              </p>
            </div>
            <Link href="/sessions">
              <Button variant="ghost" size="sm" className="rounded-xl">
                View all
                <ChevronRight className="h-4 w-4" />
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            {sessionsLoading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-16 w-full rounded-xl" />
                ))}
              </div>
            ) : recentSessions.length === 0 ? (
              <p className="text-muted-foreground py-6 text-center text-sm">
                No recent sessions. Activity will appear here.
              </p>
            ) : (
              <div className="space-y-2">
                {recentSessions.map((s) => (
                  <SessionRow key={s.id} session={s} />
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* This week at a glance — latest summary */}
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-base flex items-center gap-2">
              <FileText className="h-4 w-4" />
              This week at a glance
            </CardTitle>
          </CardHeader>
          <CardContent>
            {summariesLoading ? (
              <Skeleton className="h-24 w-full rounded-xl" />
            ) : latestSummary ? (
              <div className="rounded-xl border border-border bg-muted/30 p-4">
                <p className="text-xs text-muted-foreground">
                  {latestSummary.period_start && latestSummary.period_end
                    ? `${new Date(latestSummary.period_start).toLocaleDateString()} – ${new Date(latestSummary.period_end).toLocaleDateString()}`
                    : "Weekly summary"}
                </p>
                <p className="mt-2 text-sm leading-relaxed">{latestSummary.summary_text}</p>
              </div>
            ) : (
              <p className="text-muted-foreground py-6 text-center text-sm">
                No weekly summary yet. Check back after more activity.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Risk trend — compact */}
      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Risk trend (last 7 days)
          </CardTitle>
          {signals.length > 0 && (
            <div className="flex gap-2 flex-wrap">
              {severityCounts.map(({ severity, count }) =>
                count > 0 ? (
                  <span
                    key={severity}
                    className="rounded-lg bg-muted px-2 py-0.5 text-xs font-medium"
                  >
                    {severity}: {count}
                  </span>
                ) : null
              )}
            </div>
          )}
        </CardHeader>
        <CardContent>
          <div className="h-[100px] min-h-[100px] w-full">
            {signalsLoading ? (
              <Skeleton className="h-full w-full" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={last7Days} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} width={24} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="hsl(var(--chart-1))"
                    strokeWidth={2}
                    dot={true}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Quick navigation */}
      <div>
        <h2 className="text-sm font-medium text-muted-foreground mb-3">Quick links</h2>
        <div className="flex flex-wrap gap-3">
          <Link href="/alerts">
            <Button variant="outline" className="rounded-xl">
              Alerts
            </Button>
          </Link>
          <Link href="/sessions">
            <Button variant="outline" className="rounded-xl">
              History
            </Button>
          </Link>
          <Link href="/rings">
            <Button variant="outline" className="rounded-xl">
              Patterns
            </Button>
          </Link>
          {showAdvanced && (
            <>
              <Link href="/reports">
                <Button variant="outline" className="rounded-xl">
                  Reports
                </Button>
              </Link>
              <Link href="/agents">
                <Button variant="outline" className="rounded-xl">
                  Agent Console
                </Button>
              </Link>
              <Link href="/replay">
                <Button variant="secondary" className="rounded-xl">
                  Scenario Replay
                </Button>
              </Link>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
