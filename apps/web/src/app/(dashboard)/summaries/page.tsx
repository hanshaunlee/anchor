"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useSummaries } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { FileText } from "lucide-react";

export default function SummariesPage() {
  const { data: summaries, isLoading } = useSummaries({ limit: 20 });
  const list = summaries ?? [];

  // Use only period-scoped (weekly) summaries for the trend chart so labels and bars make sense
  const weeklyForChart = list.filter((s) => s.period_start != null).slice(0, 7);
  const chartData = weeklyForChart.map((s) => {
    const signalCount =
      s.summary_json && typeof (s.summary_json as Record<string, number>).signal_count === "number"
        ? (s.summary_json as Record<string, number>).signal_count
        : null;
    return {
      period: s.period_start
        ? new Date(s.period_start).toLocaleDateString(undefined, { month: "short", day: "numeric" })
        : "—",
      count: 1,
      // Show bar height: use signal_count when present, otherwise 1 so the trend is visible
      signals: signalCount ?? 1,
    };
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Summaries</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Weekly rollups and trend (motifs, alerts, confirmations)
        </p>
      </div>

      {!isLoading && chartData.length > 0 && (
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis dataKey="period" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} width={24} />
                  <Tooltip />
                  <Bar dataKey="signals" fill="hsl(var(--chart-1))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Weekly summaries
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-20 w-full rounded-xl" />
              ))}
            </div>
          ) : list.length === 0 ? (
            <p className="text-muted-foreground py-12 text-center">No summaries.</p>
          ) : (
            <ul className="space-y-3">
              {list.map((s) => (
                <li key={s.id} className="rounded-2xl border border-border p-4">
                  <p className="text-muted-foreground text-xs">
                    {s.period_start && s.period_end
                      ? `${new Date(s.period_start).toLocaleDateString()} – ${new Date(s.period_end).toLocaleDateString()}`
                      : "—"}
                  </p>
                  <p className="text-sm mt-1">{s.summary_text}</p>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
