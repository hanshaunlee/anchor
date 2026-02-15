"use client";

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Network, RefreshCw, ExternalLink, Download, RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";

export type TopologyHeaderProps = {
  /** Live Sync: refetch graph every 5s when ON */
  liveSync: boolean;
  onLiveSyncChange: (v: boolean) => void;
  /** Sync to Neo4j */
  onSyncNeo4j: () => void;
  syncNeo4jPending: boolean;
  neo4jEnabled: boolean;
  /** Export subgraph as JSON */
  onExportSubgraph: () => void;
  /** Reset view (fit view) */
  onResetView: () => void;
  /** Neo4j Browser link */
  neo4jBrowserUrl?: string | null;
  /** Has graph data to export */
  hasData: boolean;
};

export function TopologyHeader({
  liveSync,
  onLiveSyncChange,
  onSyncNeo4j,
  syncNeo4jPending,
  neo4jEnabled,
  onExportSubgraph,
  onResetView,
  neo4jBrowserUrl,
  hasData,
}: TopologyHeaderProps) {
  return (
    <header className="space-y-2">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
            <Network className="h-7 w-7 text-muted-foreground" />
            Household Behavioral Topology
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Structural representation of entities, sessions, and motifs. Updates as new risk signals are generated.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className={cn(
            "flex items-center gap-2 rounded-lg border px-3 py-1.5",
            liveSync ? "border-primary/50 bg-primary/5" : "border-border bg-muted/30"
          )}>
            <Switch
              id="live-sync"
              checked={liveSync}
              onCheckedChange={onLiveSyncChange}
            />
            <Label htmlFor="live-sync" className="text-sm font-medium cursor-pointer whitespace-nowrap">
              Auto-refresh {liveSync ? "ON (5s)" : "OFF"}
            </Label>
          </div>
          {neo4jEnabled && (
            <Button
              variant="outline"
              size="sm"
              onClick={onSyncNeo4j}
              disabled={syncNeo4jPending}
              className="rounded-xl"
            >
              <RefreshCw className={cn("h-4 w-4 mr-2", syncNeo4jPending && "animate-spin")} />
              Sync to Neo4j
            </Button>
          )}
          {hasData && (
            <>
              <Button variant="outline" size="sm" onClick={onExportSubgraph} className="rounded-xl">
                <Download className="h-4 w-4 mr-2" />
                Export Subgraph
              </Button>
              <Button variant="outline" size="sm" onClick={onResetView} className="rounded-xl">
                <RotateCcw className="h-4 w-4 mr-2" />
                Reset View
              </Button>
            </>
          )}
          {neo4jEnabled && neo4jBrowserUrl && (
            <a
              href={neo4jBrowserUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center rounded-xl text-sm font-medium border border-input bg-background px-4 py-2 hover:bg-accent hover:text-accent-foreground"
            >
              <ExternalLink className="h-4 w-4 mr-2" />
              Open in Neo4j Browser
            </a>
          )}
        </div>
      </div>
    </header>
  );
}
