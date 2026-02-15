"use client";

import { useRef, useEffect, useState } from "react";
import { GraphEvidence } from "@/components/graph-evidence";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";
import type { LayoutMode, NodeScaling, EdgeRenderMode } from "@/components/graph/TopologyControls";

export type ScientificGraphCanvasProps = {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  highlightNodeIds?: string[];
  resetViewKey?: number;
  pulseOnUpdate?: boolean;
  layoutMode?: LayoutMode;
  nodeScaling?: NodeScaling;
  edgeRenderMode?: EdgeRenderMode;
  className?: string;
};

function mapLayoutMode(mode: LayoutMode): "force" | "radial" {
  if (mode === "radial") return "radial";
  return "force";
}

function mapNodeScaling(s: NodeScaling): "degree" | "risk" | "static" {
  if (s === "risk" || s === "static") return s;
  return "degree";
}

function mapEdgeRender(mode: EdgeRenderMode): { edgeType: "smoothstep" | "straight" | "default"; weighted: boolean } {
  switch (mode) {
    case "straight":
      return { edgeType: "straight", weighted: false };
    case "weighted":
      return { edgeType: "smoothstep", weighted: true };
    case "curved":
      return { edgeType: "smoothstep", weighted: false };
    case "bundled":
      return { edgeType: "smoothstep", weighted: false };
    default:
      return { edgeType: "smoothstep", weighted: false };
  }
}

export function ScientificGraphCanvas({
  nodes,
  edges,
  highlightNodeIds,
  resetViewKey = 0,
  pulseOnUpdate = false,
  layoutMode = "force",
  nodeScaling = "degree",
  edgeRenderMode = "weighted",
  className,
}: ScientificGraphCanvasProps) {
  const [pulse, setPulse] = useState(false);
  const prevCountRef = useRef({ nodes: 0, edges: 0 });
  const edgeOpts = mapEdgeRender(edgeRenderMode);

  useEffect(() => {
    const prev = prevCountRef.current;
    const changed = prev.nodes !== nodes.length || prev.edges !== edges.length;
    prevCountRef.current = { nodes: nodes.length, edges: edges.length };
    if (changed && pulseOnUpdate && (nodes.length > 0 || edges.length > 0)) {
      setPulse(true);
      const t = setTimeout(() => setPulse(false), 1200);
      return () => clearTimeout(t);
    }
  }, [nodes.length, edges.length, pulseOnUpdate]);

  return (
    <div
      className={cn(
        "relative rounded-2xl border border-border overflow-hidden bg-muted/10",
        "bg-[linear-gradient(to_right,var(--border)_1px,transparent_1px),linear-gradient(to_bottom,var(--border)_1px,transparent_1px)] bg-[size:12px_12px]",
        pulse && "animate-in [--tw-enter-opacity:0.7] duration-300",
        className
      )}
    >
      <div className="h-[520px] w-full relative z-0">
        <GraphEvidence
          key={resetViewKey}
          variant="graph"
          nodes={nodes}
          edges={edges}
          highlightIds={highlightNodeIds}
          backgroundVariant="scientific"
          layoutMode={mapLayoutMode(layoutMode)}
          nodeScaling={mapNodeScaling(nodeScaling)}
          edgeType={edgeOpts.edgeType}
          edgeWeighted={edgeOpts.weighted}
        />
      </div>
    </div>
  );
}
