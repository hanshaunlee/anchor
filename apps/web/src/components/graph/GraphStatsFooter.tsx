"use client";

import { cn } from "@/lib/utils";

export type GraphStatsFooterProps = {
  nodeCount: number;
  edgeCount: number;
  density: number;
  ringsCount?: number | null;
  watchlistedCount?: number | null;
  driftStatus?: string | null;
  connectedComponents?: number | null;
  largestComponentSize?: number | null;
  className?: string;
};

/** Scientific / evidence-density style: label + monospace value pairs. */
export function GraphStatsFooter({
  nodeCount,
  edgeCount,
  density,
  ringsCount,
  watchlistedCount,
  driftStatus,
  connectedComponents,
  largestComponentSize,
  className,
}: GraphStatsFooterProps) {
  const rows: { label: string; value: string }[] = [
    { label: "Nodes", value: String(nodeCount) },
    { label: "Edges", value: String(edgeCount) },
    { label: "Density", value: density.toFixed(3) },
  ];
  if (connectedComponents != null) rows.push({ label: "Components", value: String(connectedComponents) });
  if (largestComponentSize != null) rows.push({ label: "Largest", value: String(largestComponentSize) });
  if (ringsCount != null) rows.push({ label: "Rings", value: String(ringsCount) });
  if (watchlistedCount != null) rows.push({ label: "Watchlisted", value: String(watchlistedCount) });
  if (driftStatus != null) rows.push({ label: "Drift", value: driftStatus });

  return (
    <footer
      className={cn(
        "sticky bottom-0 z-10 flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-border bg-muted/20 px-4 py-2.5 text-xs",
        className
      )}
    >
      {rows.map((r, i) => (
        <span key={i} className="flex items-baseline gap-1.5">
          <span className="text-muted-foreground uppercase tracking-wider">{r.label}</span>
          <span className="font-mono tabular-nums text-foreground">{r.value}</span>
        </span>
      ))}
    </footer>
  );
}
