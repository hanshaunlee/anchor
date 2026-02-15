"use client";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { SlidersHorizontal } from "lucide-react";

export type LayoutMode = "force" | "radial" | "community" | "ring_only" | "motif";
export type NodeScaling = "degree" | "betweenness" | "risk" | "static";
export type EdgeRenderMode = "curved" | "straight" | "bundled" | "weighted";

export type TopologyControlsProps = {
  onlySuspicious: boolean;
  onOnlySuspiciousChange: (v: boolean) => void;
  onlyRecent7d: boolean;
  onOnlyRecent7dChange: (v: boolean) => void;
  highlightRings: boolean;
  onHighlightRingsChange: (v: boolean) => void;
  highlightMotifs: boolean;
  onHighlightMotifsChange: (v: boolean) => void;
  highlightAlertNodes: boolean;
  onHighlightAlertNodesChange: (v: boolean) => void;
  layoutMode: LayoutMode;
  onLayoutModeChange: (v: LayoutMode) => void;
  nodeScaling: NodeScaling;
  onNodeScalingChange: (v: NodeScaling) => void;
  edgeRenderMode: EdgeRenderMode;
  onEdgeRenderModeChange: (v: EdgeRenderMode) => void;
};

const LAYOUT_LABELS: Record<LayoutMode, string> = {
  force: "Force-directed",
  radial: "Radial cluster",
  community: "Community",
  ring_only: "Ring only",
  motif: "Motif view",
};

const SCALING_LABELS: Record<NodeScaling, string> = {
  degree: "Degree",
  betweenness: "Betweenness",
  risk: "Risk contribution",
  static: "Static",
};

const EDGE_LABELS: Record<EdgeRenderMode, string> = {
  curved: "Curved",
  straight: "Straight",
  bundled: "Bundled",
  weighted: "Weighted",
};

export function TopologyControls({
  onlySuspicious,
  onOnlySuspiciousChange,
  onlyRecent7d,
  onOnlyRecent7dChange,
  highlightRings,
  onHighlightRingsChange,
  highlightMotifs,
  onHighlightMotifsChange,
  highlightAlertNodes,
  onHighlightAlertNodesChange,
  layoutMode,
  onLayoutModeChange,
  nodeScaling,
  onNodeScalingChange,
  edgeRenderMode,
  onEdgeRenderModeChange,
}: TopologyControlsProps) {
  return (
    <div className="rounded-xl border border-border bg-muted/20 px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <SlidersHorizontal className="h-4 w-4" />
          Topology controls
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2">
          <Label className="text-xs font-medium text-muted-foreground">Filters:</Label>
          <Button
            variant={onlySuspicious ? "secondary" : "ghost"}
            size="sm"
            className="rounded-lg h-7 text-xs"
            onClick={() => onOnlySuspiciousChange(!onlySuspicious)}
          >
            Suspicious only
          </Button>
          <Button
            variant={onlyRecent7d ? "secondary" : "ghost"}
            size="sm"
            className="rounded-lg h-7 text-xs"
            onClick={() => onOnlyRecent7dChange(!onlyRecent7d)}
          >
            Last 7 days
          </Button>
          <Button
            variant={highlightRings ? "secondary" : "ghost"}
            size="sm"
            className="rounded-lg h-7 text-xs"
            onClick={() => onHighlightRingsChange(!highlightRings)}
          >
            Ring clusters
          </Button>
          <Button
            variant={highlightMotifs ? "secondary" : "ghost"}
            size="sm"
            className="rounded-lg h-7 text-xs"
            onClick={() => onHighlightMotifsChange(!highlightMotifs)}
          >
            Motif nodes
          </Button>
          <Button
            variant={highlightAlertNodes ? "secondary" : "ghost"}
            size="sm"
            className="rounded-lg h-7 text-xs"
            onClick={() => onHighlightAlertNodesChange(!highlightAlertNodes)}
          >
            Alert nodes
          </Button>
        </div>

        {/* Layout */}
        <div className="flex flex-wrap items-center gap-2">
          <Label className="text-xs font-medium text-muted-foreground">Layout:</Label>
          {(Object.keys(LAYOUT_LABELS) as LayoutMode[]).map((mode) => (
            <Button
              key={mode}
              variant={layoutMode === mode ? "secondary" : "ghost"}
              size="sm"
              className="rounded-lg h-7 text-xs"
              onClick={() => onLayoutModeChange(mode)}
            >
              {LAYOUT_LABELS[mode]}
            </Button>
          ))}
        </div>

        {/* Node scaling */}
        <div className="flex flex-wrap items-center gap-2">
          <Label className="text-xs font-medium text-muted-foreground">Node size:</Label>
          {(Object.keys(SCALING_LABELS) as NodeScaling[]).map((s) => (
            <Button
              key={s}
              variant={nodeScaling === s ? "secondary" : "ghost"}
              size="sm"
              className="rounded-lg h-7 text-xs"
              onClick={() => onNodeScalingChange(s)}
            >
              {SCALING_LABELS[s]}
            </Button>
          ))}
        </div>

        {/* Edge rendering */}
        <div className="flex flex-wrap items-center gap-2">
          <Label className="text-xs font-medium text-muted-foreground">Edges:</Label>
          {(Object.keys(EDGE_LABELS) as EdgeRenderMode[]).map((e) => (
            <Button
              key={e}
              variant={edgeRenderMode === e ? "secondary" : "ghost"}
              size="sm"
              className="rounded-lg h-7 text-xs"
              onClick={() => onEdgeRenderModeChange(e)}
            >
              {EDGE_LABELS[e]}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
