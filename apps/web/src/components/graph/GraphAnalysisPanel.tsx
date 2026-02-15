"use client";

import { useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ChevronRight, ChevronDown, BarChart3, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GraphStructuralMetrics } from "@/lib/graph-metrics";
import type { RiskSignalCard } from "@/lib/api/schemas";

export type GraphAnalysisPanelProps = {
  /** Structural metrics (live) */
  metrics: GraphStructuralMetrics | null;
  /** Node id -> label for top central nodes */
  nodeLabels: Map<string, string>;
  /** Active alerts (open or recent); clicking one can highlight contributing nodes */
  activeAlerts: RiskSignalCard[];
  onHighlightAlert?: (signalId: string) => void;
  /** Default expanded */
  defaultOpen?: boolean;
  className?: string;
};

export function GraphAnalysisPanel({
  metrics,
  nodeLabels,
  activeAlerts,
  onHighlightAlert,
  defaultOpen = false,
  className,
}: GraphAnalysisPanelProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={cn("rounded-xl border border-border bg-card overflow-hidden", className)}>
      <button
        type="button"
        className="w-full flex items-center gap-2 p-4 text-left hover:bg-muted/30 transition"
        onClick={() => setOpen(!open)}
      >
        <BarChart3 className="h-5 w-5 text-muted-foreground" />
        <span className="font-semibold">Analysis panel</span>
        {open ? <ChevronDown className="h-4 w-4 ml-auto" /> : <ChevronRight className="h-4 w-4 ml-auto" />}
      </button>
      {open && (
        <div className="border-t border-border p-4 space-y-4">
          {metrics ? (
            <>
              <section>
                <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                  Live structural metrics
                </h4>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                  <dt className="text-muted-foreground">Nodes</dt>
                  <dd className="font-medium tabular-nums">{metrics.nodeCount}</dd>
                  <dt className="text-muted-foreground">Edges</dt>
                  <dd className="font-medium tabular-nums">{metrics.edgeCount}</dd>
                  <dt className="text-muted-foreground">Components</dt>
                  <dd className="font-medium tabular-nums">{metrics.connectedComponents}</dd>
                  <dt className="text-muted-foreground">Largest component</dt>
                  <dd className="font-medium tabular-nums">{metrics.largestComponentSize}</dd>
                  <dt className="text-muted-foreground">Density</dt>
                  <dd className="font-medium tabular-nums">{metrics.density.toFixed(2)}</dd>
                  <dt className="text-muted-foreground">Avg. clustering</dt>
                  <dd className="font-medium tabular-nums">{metrics.avgClusteringCoefficient.toFixed(2)}</dd>
                </dl>
              </section>
              {metrics.topCentralNodeIds.length > 0 && (
                <section>
                  <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                    Top central nodes
                  </h4>
                  <ul className="text-sm space-y-1">
                    {metrics.topCentralNodeIds.map((id) => (
                      <li key={id} className="truncate" title={id}>
                        {nodeLabels.get(id) || id.slice(0, 8)}…
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Load graph to see metrics.</p>
          )}

          <section>
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" />
              Active alerts
            </h4>
            {activeAlerts.length === 0 ? (
              <p className="text-sm text-muted-foreground">No open alerts.</p>
            ) : (
              <ul className="space-y-2">
                {activeAlerts.slice(0, 10).map((s) => (
                  <li key={s.id} className="flex items-center gap-2">
                    <Link
                      href={`/alerts/${s.id}`}
                      className="text-sm font-medium text-primary hover:underline truncate flex-1 min-w-0"
                    >
                      {s.id.slice(0, 8)}… · S{s.severity}
                    </Link>
                    {onHighlightAlert && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="rounded-lg h-7 text-xs shrink-0"
                        onClick={() => onHighlightAlert(s.id)}
                      >
                        Highlight
                      </Button>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
