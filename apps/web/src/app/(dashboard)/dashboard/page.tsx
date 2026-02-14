"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useRiskSignals, useHouseholdMe } from "@/hooks/use-api";
import { useRiskSignalStream } from "@/hooks/use-risk-signal-stream";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, TrendingUp } from "lucide-react";
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

export default function DashboardPage() {
  const { data: me } = useHouseholdMe();
  const { data: signalsData, isLoading } = useRiskSignals({ limit: 10 });
  useRiskSignalStream(!!me && me.role !== "elder");

  const signals = useMemo(() => signalsData?.signals ?? [], [signalsData?.signals]);
  const openCount = signals.filter((s) => s.status === "open").length;
  const severityCounts = useMemo(() => {
    const c: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    signals.forEach((s) => {
      c[s.severity] = (c[s.severity] ?? 0) + 1;
    });
    return Object.entries(c).map(([sev, count]) => ({ severity: `S${sev}`, count }));
  }, [signals]);

  const last7Days = useMemo(() => {
    const d: { day: string; score: number; count: number }[] = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dayStr = date.toISOString().slice(0, 10);
      const daySignals = signals.filter(
        (s) => new Date(s.ts).toISOString().slice(0, 10) === dayStr
      );
      const maxScore = daySignals.length ? Math.max(...daySignals.map((s) => s.score)) : 0;
      d.push({ day: dayStr.slice(5), score: maxScore, count: daySignals.length });
    }
    return d;
  }, [signals]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Today</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Latest risk signals and quick actions
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Open alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-semibold">{openCount}</p>
            )}
          </CardContent>
        </Card>
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Severity distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-full" />
            ) : (
              <div className="flex gap-2 flex-wrap">
                {severityCounts.map(({ severity, count }) =>
                  count > 0 ? (
                    <span
                      key={severity}
                      className="rounded-lg bg-muted px-2 py-1 text-xs font-medium"
                    >
                      {severity}: {count}
                    </span>
                  ) : null
                )}
                {signals.length === 0 && (
                  <span className="text-muted-foreground text-sm">None</span>
                )}
              </div>
            )}
          </CardContent>
        </Card>
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Risk (7 days)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[80px] min-h-[80px] w-full">
              {isLoading ? (
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
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Latest risk signals</CardTitle>
          <p className="text-muted-foreground text-sm">Review and investigate</p>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 w-full rounded-xl" />
              ))}
            </div>
          ) : signals.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">
              No risk signals in the selected period.
            </p>
          ) : (
            <div className="space-y-3">
              {signals.slice(0, 5).map((s) => (
                <RiskSignalCard key={s.id} signal={s} />
              ))}
              <Link href="/alerts" className="block">
                <Button variant="outline" className="w-full rounded-xl">
                  View all alerts
                </Button>
              </Link>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex flex-wrap gap-3">
        <Link href="/alerts">
          <Button variant="outline" className="rounded-xl">Alerts</Button>
        </Link>
        <Link href="/rings">
          <Button variant="outline" className="rounded-xl">Rings</Button>
        </Link>
        <Link href="/reports">
          <Button variant="outline" className="rounded-xl">Reports</Button>
        </Link>
        <Link href="/replay">
          <Button variant="secondary" className="rounded-xl">Scenario Replay</Button>
        </Link>
        <Link href="/agents">
          <Button variant="outline" className="rounded-xl">Agents</Button>
        </Link>
      </div>
    </div>
  );
}
