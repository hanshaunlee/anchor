"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { ArrowLeft, ShieldAlert, Play } from "lucide-react";

type RedteamReport = Awaited<ReturnType<typeof api.getRedteamReport>>;

export default function RedteamReportPage() {
  const [report, setReport] = useState<RedteamReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getRedteamReport();
      setReport(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load red-team report");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="space-y-4">
        <div className="flex gap-2">
          <Link href="/reports">
            <Button variant="ghost" size="sm" className="gap-2">Reports</Button>
          </Link>
          <Link href="/agents">
            <Button variant="ghost" size="sm" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Agents
            </Button>
          </Link>
        </div>
        <Card className="rounded-2xl">
          <CardContent className="pt-6">
            <p className="text-destructive">{error ?? "No red-team report found. Run the Synthetic Red-Team agent first."}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const summary = (report.summary_json ?? {}) as Record<string, unknown>;
  const passRate = summary.regression_pass_rate as number | undefined;
  const scenariosGenerated = summary.scenarios_generated as number | undefined;
  const failingCases = (summary.failing_cases ?? []) as Array<{ scenario_id?: string; assertion?: string; expected?: string; got?: string }>;
  const regressionPassed = summary.regression_passed as boolean | undefined;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/reports">
          <Button variant="ghost" size="sm">Reports</Button>
        </Link>
        <Link href="/agents">
          <Button variant="ghost" size="sm" className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Agents
          </Button>
        </Link>
        <span className="text-muted-foreground text-sm">
          {report.started_at ? new Date(report.started_at).toLocaleString() : ""}
        </span>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <ShieldAlert className="h-5 w-5" />
            Red-team regression report
          </CardTitle>
          <p className="text-muted-foreground text-sm">{summary.headline as string}</p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="rounded-xl border border-border px-4 py-2">
              <span className="text-xs font-medium text-muted-foreground">Pass rate</span>
              <p className="text-xl font-semibold">{passRate != null ? `${(passRate * 100).toFixed(0)}%` : "—"}</p>
            </div>
            <div className="rounded-xl border border-border px-4 py-2">
              <span className="text-xs font-medium text-muted-foreground">Scenarios</span>
              <p className="text-xl font-semibold">{scenariosGenerated ?? "—"}</p>
            </div>
            {regressionPassed !== undefined && (
              <span className={regressionPassed ? "text-green-600 dark:text-green-400" : "text-destructive"}>
                {regressionPassed ? "Passed" : "Below threshold"}
              </span>
            )}
          </div>
          {failingCases.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">Failing cases ({failingCases.length})</p>
              <ul className="space-y-1 text-sm max-h-48 overflow-y-auto">
                {failingCases.slice(0, 20).map((f, i) => (
                  <li key={i} className="rounded bg-muted/50 px-2 py-1">
                    {f.scenario_id ?? "?"}: {f.assertion ?? ""} — expected {f.expected ?? ""}, got {String(f.got ?? "")}
                  </li>
                ))}
                {failingCases.length > 20 && <li className="text-muted-foreground text-xs">… and {failingCases.length - 20} more</li>}
              </ul>
            </div>
          )}
          <Link href="/replay">
            <Button variant="outline" size="sm" className="gap-2">
              <Play className="h-4 w-4" />
              Open in replay
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
