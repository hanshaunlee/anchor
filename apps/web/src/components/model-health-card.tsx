"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Activity, ExternalLink } from "lucide-react";

export type ModelHealthCardProps = {
  /** From GET /agents/status for agent_name=model_health */
  lastRun: {
    last_run_id?: string | null;
    last_run_at?: string | null;
    last_run_status?: string | null;
    last_run_summary?: Record<string, unknown> | null;
  } | null;
  /** Whether current user can run maintenance */
  canRun?: boolean;
  onRunMaintenance?: () => void;
  isRunning?: boolean;
};

function statusColor(status: string | null | undefined): "default" | "secondary" | "destructive" {
  const s = (status ?? "").toLowerCase();
  if (s === "ok" || s === "completed" || s === "stable") return "default";
  if (s === "warning" || s === "recalibrate" || s === "partial") return "secondary";
  return "destructive";
}

export function ModelHealthCard({
  lastRun,
  canRun,
  onRunMaintenance,
  isRunning,
}: ModelHealthCardProps) {
  const summary = (lastRun?.last_run_summary ?? {}) as Record<string, unknown>;
  const driftDetected = summary.drift_detected as boolean | undefined;
  const recommendation = summary.recommendation as string | undefined;
  const eceBefore = summary.ece_before as number | undefined;
  const eceAfter = summary.ece_after as number | undefined;
  const coverage = summary.conformal_coverage as number | undefined;
  const centroidShift = summary.centroid_shift as number | undefined;
  const mmdRbf = summary.mmd_rbf as number | undefined;
  const labelCount = summary.label_count as number | undefined;
  const insufficientSamples = summary.insufficient_samples as boolean | undefined;

  const overallStatus =
    insufficientSamples ? "Insufficient samples" :
    driftDetected === true ? "Investigate drift" :
    recommendation && recommendation.toLowerCase() !== "stable" ? recommendation : "Stable";
  const pillVariant = statusColor(
    overallStatus === "Stable" ? "ok" : overallStatus === "Insufficient samples" ? "warning" : "recalibrate"
  );

  return (
    <Card className="rounded-2xl shadow-sm border-muted/50">
      <CardHeader className="pb-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Model Health
          </CardTitle>
          <Badge variant={pillVariant} className="rounded-lg">
            {overallStatus}
          </Badge>
          {lastRun?.last_run_at && (
            <span className="text-muted-foreground text-sm">
              Last run: {new Date(lastRun.last_run_at).toLocaleString()}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {!lastRun && (
          <p className="text-muted-foreground text-sm">No model health run yet. Run maintenance to check drift and calibration.</p>
        )}
        {lastRun && (
          <>
            {(driftDetected != null || centroidShift != null || mmdRbf != null) && (
              <div className="rounded-xl bg-muted/40 px-4 py-3 text-sm space-y-1">
                <p className="font-medium">Drift</p>
                {insufficientSamples && <p className="text-muted-foreground">Insufficient samples for drift metrics.</p>}
                {!insufficientSamples && (
                  <>
                    {driftDetected != null && <p>Detected: {String(driftDetected)}</p>}
                    {centroidShift != null && <p>Centroid shift: {Number(centroidShift).toFixed(4)}</p>}
                    {mmdRbf != null && <p>MMD (RBF): {Number(mmdRbf).toFixed(4)}</p>}
                  </>
                )}
              </div>
            )}
            {(eceBefore != null || eceAfter != null || labelCount != null) && (
              <div className="rounded-xl bg-muted/40 px-4 py-3 text-sm space-y-1">
                <p className="font-medium">Calibration</p>
                {eceBefore != null && <p>ECE before: {(eceBefore * 100).toFixed(2)}%</p>}
                {eceAfter != null && <p>ECE after: {(eceAfter * 100).toFixed(2)}%</p>}
                {labelCount != null && <p>Label count: {labelCount}</p>}
              </div>
            )}
            {coverage != null && (
              <div className="rounded-xl bg-muted/40 px-4 py-3 text-sm">
                <p className="font-medium">Conformal validity</p>
                <p>Coverage: {(coverage * 100).toFixed(2)}%</p>
              </div>
            )}
            {recommendation && (
              <p className="text-sm text-muted-foreground">
                Recommendation: {recommendation}
              </p>
            )}
            {lastRun.last_run_id && (
              <p className="text-xs text-muted-foreground">
                Run ID: {lastRun.last_run_id}
              </p>
            )}
          </>
        )}
        {canRun && (
          <Button
            variant="outline"
            className="rounded-xl"
            onClick={onRunMaintenance}
            disabled={isRunning}
          >
            {isRunning ? "Running…" : "Run maintenance now"}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
