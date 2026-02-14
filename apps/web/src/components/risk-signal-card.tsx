"use client";

import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { RiskSignalCard as RiskSignalCardType } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";
import { ChevronRight } from "lucide-react";

const statusVariant: Record<string, string> = {
  open: "bg-status-open/15 text-status-open border-status-open/30",
  acknowledged: "bg-status-acknowledged/15 text-status-acknowledged border-status-acknowledged/30",
  dismissed: "bg-status-dismissed/15 text-status-dismissed border-status-dismissed/30",
  escalated: "bg-status-escalated/15 text-status-escalated border-status-escalated/30",
};

const severityColor: Record<number, string> = {
  1: "bg-severity-1/20 text-severity-1",
  2: "bg-severity-2/20 text-severity-2",
  3: "bg-severity-3/20 text-severity-3",
  4: "bg-severity-4/20 text-severity-4",
  5: "bg-severity-5/20 text-severity-5",
};

export function RiskSignalCard({ signal }: { signal: RiskSignalCardType }) {
  const time = new Date(signal.ts).toLocaleString(undefined, {
    dateStyle: "short",
    timeStyle: "short",
  });
  const topReasons = signal.summary
    ? signal.summary.slice(0, 80) + (signal.summary.length > 80 ? "…" : "")
    : signal.signal_type;

  return (
    <Card className="rounded-2xl shadow-sm transition hover:shadow-md">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                className={cn("rounded-lg border", statusVariant[signal.status] ?? "bg-muted")}
                variant="outline"
              >
                {signal.status}
              </Badge>
              <span
                className={cn(
                  "rounded-lg px-2 py-0.5 text-xs font-medium",
                  severityColor[signal.severity] ?? "bg-muted"
                )}
              >
                Severity {signal.severity}
              </span>
              <span className="text-muted-foreground text-xs">{signal.signal_type}</span>
            </div>
            <p className="text-sm font-medium leading-snug">
              Score: {(signal.score * 100).toFixed(0)}%
            </p>
            <p className="text-muted-foreground text-xs leading-snug line-clamp-2">
              {topReasons}
            </p>
            <p className="text-muted-foreground text-xs">{time}</p>
          </div>
          <Link href={`/alerts/${signal.id}`}>
            <Button size="sm" variant="outline" className="rounded-xl shrink-0">
              Investigate
              <ChevronRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
