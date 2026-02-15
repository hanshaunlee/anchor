"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ExplainableIds } from "@/components/explainable-ids";
import { api } from "@/lib/api";
import { ArrowLeft, FileText } from "lucide-react";

type Report = Awaited<ReturnType<typeof api.getNarrativeReport>>;

export default function NarrativeReportPage({ params }: { params: { id: string } }) {
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!params.id) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getNarrativeReport(params.id);
      setReport(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load report");
    } finally {
      setLoading(false);
    }
  }, [params.id]);

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
            <p className="text-destructive">{error ?? "Report not found."}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const payload = report.report_json ?? {};
  const reports = (payload.reports as Report["report_json"]["reports"]) ?? [];

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
          {report.created_at ? new Date(report.created_at).toLocaleString() : ""}
        </span>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <FileText className="h-5 w-5" />
            Evidence Narrative Report
          </CardTitle>
          {payload.headline && (
            <p className="text-muted-foreground text-sm font-medium">{payload.headline}</p>
          )}
          {payload.narrative_preview && (
            <p className="text-sm text-foreground/90">{payload.narrative_preview}</p>
          )}
        </CardHeader>
        <CardContent className="space-y-6">
          {reports.length === 0 && (
            <p className="text-muted-foreground text-sm">No per-signal reports in this run.</p>
          )}
          {reports.length > 0 && (
            <ExplainableIds
              context="alert_ids"
              items={reports.filter((r) => r.signal_id).map((r) => ({ id: String(r.signal_id), label: null }))}
              title="What these signals are"
              className="mb-4"
            />
          )}
          {reports.map((r, i) => (
            <div key={r.signal_id ?? i} className="rounded-xl border border-border bg-muted/30 p-4 space-y-3">
              <p className="text-xs font-medium text-muted-foreground">
                Signal: {String(r.signal_id).slice(0, 8)}…
              </p>
              {r.caregiver_narrative && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">Caregiver narrative</p>
                  <p className="text-sm">{r.caregiver_narrative.summary || r.caregiver_narrative.headline}</p>
                  {(r.caregiver_narrative.recommended_next_steps?.length ?? 0) > 0 && (
                    <ul className="mt-2 list-disc list-inside text-xs text-muted-foreground">
                      {r.caregiver_narrative.recommended_next_steps?.map((s, j) => (
                        <li key={j}>{s}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              {r.elder_safe && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">Elder-safe</p>
                  <p className="text-sm">{r.elder_safe.plain_language_summary}</p>
                  {(r.elder_safe.do_now_checklist?.length ?? 0) > 0 && (
                    <ul className="mt-2 list-disc list-inside text-xs">
                      {r.elder_safe.do_now_checklist?.map((s, j) => (
                        <li key={j}>{s}</li>
                      ))}
                    </ul>
                  )}
                  {r.elder_safe.reassurance_line && (
                    <p className="mt-2 text-xs italic text-muted-foreground">{r.elder_safe.reassurance_line}</p>
                  )}
                </div>
              )}
              {(r.hypotheses?.length ?? 0) > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-1">Hypotheses</p>
                  <ul className="list-disc list-inside text-xs space-y-1">
                    {(r.hypotheses as Array<{ title?: string }>).map((h, j) => (
                      <li key={j}>{h.title ?? String(h)}</li>
                    ))}
                  </ul>
                </div>
              )}
              {r.signal_id && (
                <Link href={`/alerts/${r.signal_id}`}>
                  <Button variant="outline" size="sm" className="text-xs">
                    View alert
                  </Button>
                </Link>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
