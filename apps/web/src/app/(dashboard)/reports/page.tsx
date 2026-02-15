"use client";

import Link from "next/link";
import { ReportMiniCard } from "@/components/protection/ReportMiniCard";
import { useProtectionReports } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import { FileText, TrendingDown, ShieldAlert, ChevronRight } from "lucide-react";

const REPORT_ENTRIES = [
  {
    kind: "Evidence Narrative",
    apiKind: "Narrative",
    description: "Caregiver narrative, elder-safe view, and hypotheses per signal. Run the Evidence Narrative agent, then use \"View report\" on the Automation page to open the latest report.",
    href: "/agents",
    buttonLabel: "Go to Automation",
    icon: <FileText className="h-4 w-4" />,
  },
  {
    kind: "Calibration",
    apiKind: "Calibration",
    description: "Before/after ECE, precision and recall from the Continual Calibration agent. Opens the latest run's report.",
    href: "/reports/calibration",
    buttonLabel: "View calibration report",
    icon: <TrendingDown className="h-4 w-4" />,
  },
  {
    kind: "Red-team",
    apiKind: "Redteam",
    description: "Regression pass rate, failing cases, and \"Open in replay\" from the Synthetic Red-Team agent. Opens the latest run's report.",
    href: "/reports/redteam",
    buttonLabel: "View red-team report",
    icon: <ShieldAlert className="h-4 w-4" />,
  },
] as const;

export default function ReportsPage() {
  const { data: reports, isLoading } = useProtectionReports();

  const byKind: Record<string, { last_run_at?: string | null; last_run_id?: string | null; summary?: string | null; status?: string | null }> = {};
  if (reports) {
    for (const r of reports) {
      byKind[r.kind] = {
        last_run_at: r.last_run_at,
        last_run_id: r.last_run_id,
        summary: r.summary,
        status: r.status,
      };
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Reports</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Agent-generated reports. Run the corresponding agent from the Automation page to produce data, then open the report below.
        </p>
        <Link href="/protection" className="text-xs text-primary hover:underline mt-1 inline-flex items-center gap-0.5">
          View on Protection
          <ChevronRight className="h-3 w-3" />
        </Link>
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-48 rounded-2xl" />
          ))}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {REPORT_ENTRIES.map((entry) => {
            const run = byKind[entry.apiKind];
            return (
              <ReportMiniCard
                key={entry.kind}
                kind={entry.kind}
                description={entry.description}
                lastRunAt={run?.last_run_at}
                lastRunId={run?.last_run_id}
                summary={run?.summary ?? undefined}
                status={run?.status ?? undefined}
                href={entry.href}
                buttonLabel={entry.buttonLabel}
                icon={entry.icon}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
