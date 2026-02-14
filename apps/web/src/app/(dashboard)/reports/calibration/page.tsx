"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { ArrowLeft, TrendingDown } from "lucide-react";

type CalibrationReport = Awaited<ReturnType<typeof api.getCalibrationReport>>;

export default function CalibrationReportPage() {
  const [report, setReport] = useState<CalibrationReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getCalibrationReport();
      setReport(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load calibration report");
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
            <p className="text-destructive">{error ?? "No calibration report found. Run the Continual Calibration agent first."}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const summary = (report.summary_json ?? {}) as Record<string, unknown>;
  const metrics = (summary.key_metrics ?? {}) as Record<string, unknown>;
  const calReport = (summary.calibration_report ?? {}) as Record<string, unknown>;
  const beforeEce = calReport.before_ece ?? metrics.before_ece;
  const afterEce = calReport.after_ece ?? metrics.after_ece;
  const feedbackCount = summary.feedback_count ?? calReport.feedback_count;
  const precisionRecallBefore = calReport.precision_recall_before as { precision?: number; recall?: number } | undefined;
  const precisionRecallAfter = calReport.precision_recall_after as { precision?: number; recall?: number } | undefined;

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
            <TrendingDown className="h-5 w-5" />
            Calibration report
          </CardTitle>
          <p className="text-muted-foreground text-sm">{summary.headline as string}</p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-2 gap-4 max-w-md">
            <div className="rounded-xl border border-border p-4">
              <p className="text-xs font-medium text-muted-foreground mb-1">ECE before</p>
              <p className="text-2xl font-semibold">{beforeEce != null ? Number(beforeEce).toFixed(3) : "—"}</p>
            </div>
            <div className="rounded-xl border border-border p-4">
              <p className="text-xs font-medium text-muted-foreground mb-1">ECE after</p>
              <p className="text-2xl font-semibold text-primary">{afterEce != null ? Number(afterEce).toFixed(3) : "—"}</p>
            </div>
          </div>
          {feedbackCount != null && (
            <p className="text-sm text-muted-foreground">Feedback labels used: {Number(feedbackCount)}</p>
          )}
          {(precisionRecallBefore || precisionRecallAfter) && (
            <div className="text-sm space-y-2">
              {precisionRecallBefore && (
                <p>Precision/recall before: P={Number(precisionRecallBefore.precision).toFixed(2)} R={Number(precisionRecallBefore.recall).toFixed(2)}</p>
              )}
              {precisionRecallAfter && (
                <p>Precision/recall after: P={Number(precisionRecallAfter.precision).toFixed(2)} R={Number(precisionRecallAfter.recall).toFixed(2)}</p>
              )}
            </div>
          )}
          {(summary.key_findings as string[])?.length ? (
            <ul className="list-disc list-inside text-sm text-muted-foreground">
              {(summary.key_findings as string[]).map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
