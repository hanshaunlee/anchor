"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRiskSignals } from "@/hooks/use-api";
import { useRiskSignalStream } from "@/hooks/use-risk-signal-stream";
import { useHouseholdMe } from "@/hooks/use-api";
import { RiskSignalCard } from "@/components/risk-signal-card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

export default function AlertsPage() {
  const [status, setStatus] = useState<string>("");
  const [severityMin, setSeverityMin] = useState<number | undefined>(undefined);
  const { data: me } = useHouseholdMe();
  const { data, isLoading } = useRiskSignals({
    status: status || undefined,
    severityMin,
    limit: 50,
  });
  useRiskSignalStream(!!me && me.role !== "elder");

  const signals = data?.signals ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Alerts</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Risk signals with filters. Updates in real time when new signals arrive.
        </p>
      </div>

      <div className="flex flex-wrap gap-4">
        <div className="space-y-2">
          <Label className="text-xs">Status</Label>
          <Select value={status || "all"} onValueChange={(v) => setStatus(v === "all" ? "" : v)}>
            <SelectTrigger className="w-[160px] rounded-xl">
              <SelectValue placeholder="All" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="open">Open</SelectItem>
              <SelectItem value="acknowledged">Acknowledged</SelectItem>
              <SelectItem value="dismissed">Dismissed</SelectItem>
              <SelectItem value="escalated">Escalated</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label className="text-xs">Min severity</Label>
          <Select
            value={severityMin?.toString() ?? "all"}
            onValueChange={(v) => setSeverityMin(v === "all" ? undefined : parseInt(v, 10))}
          >
            <SelectTrigger className="w-[120px] rounded-xl">
              <SelectValue placeholder="Any" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Any</SelectItem>
              {[1, 2, 3, 4, 5].map((s) => (
                <SelectItem key={s} value={String(s)}>
                  {s}+
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            Risk signals ({data?.total ?? 0})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-28 w-full rounded-xl" />
              ))}
            </div>
          ) : signals.length === 0 ? (
            <p className="text-muted-foreground py-12 text-center">
              No risk signals match the filters.
            </p>
          ) : (
            <div className="space-y-3">
              {signals.map((s) => (
                <RiskSignalCard key={s.id} signal={s} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
