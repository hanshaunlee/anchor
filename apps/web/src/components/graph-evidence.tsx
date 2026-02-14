"use client";

import { useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  Panel,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";

const NODE_COLORS: Record<string, string> = {
  person: "hsl(var(--chart-1))",
  entity: "hsl(var(--chart-2))",
  device: "hsl(var(--chart-3))",
  utterance: "hsl(var(--chart-4))",
  intent: "hsl(var(--chart-5))",
  default: "hsl(var(--muted-foreground))",
};

function toFlowNodes(nodes: SubgraphNode[]): Node[] {
  const positions: Record<string, { x: number; y: number }> = {};
  const radius = 180;
  nodes.forEach((n, i) => {
    const angle = (i / Math.max(nodes.length, 1)) * 2 * Math.PI - Math.PI / 2;
    positions[n.id] = {
      x: 250 + radius * Math.cos(angle),
      y: 200 + radius * Math.sin(angle),
    };
  });
  return nodes.map((n) => ({
    id: n.id,
    type: "default",
    position: positions[n.id] ?? { x: 0, y: 0 },
    data: {
      label: n.label || n.id,
      score: n.score,
      nodeType: n.type,
    },
    style: {
      background: NODE_COLORS[n.type] ?? NODE_COLORS.default,
      color: "white",
      border: "2px solid transparent",
      borderRadius: 12,
      padding: "8px 12px",
      fontSize: 11,
    },
  }));
}

function toFlowEdges(edges: SubgraphEdge[]): Edge[] {
  return edges.map((e) => ({
    id: `${e.src}-${e.dst}`,
    source: e.src,
    target: e.dst,
    type: "smoothstep",
    animated: false,
    style: { strokeWidth: Math.max(1, (e.weight ?? 0.5) * 3) },
    markerEnd: MarkerType.ArrowClosed,
  }));
}

export function GraphEvidence({
  nodes,
  edges,
  highlightIds,
  onPlayPath,
  className,
}: {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  highlightIds?: string[];
  onPlayPath?: () => void;
  className?: string;
}) {
  const initialNodes = useMemo(() => toFlowNodes(nodes), [nodes]);
  const initialEdges = useMemo(() => toFlowEdges(edges), [edges]);
  const [flowNodes, , onNodesChange] = useNodesState(initialNodes);
  const [flowEdges, , onEdgesChange] = useEdgesState(initialEdges);

  const highlighted = useMemo(() => new Set(highlightIds ?? []), [highlightIds]);

  const nodeTypes = useMemo(() => ({}), []);
  const fitViewOptions = useMemo(() => ({ padding: 0.2, maxZoom: 1.2 }), []);

  return (
    <div className={cn("rounded-2xl border border-border bg-card overflow-hidden", className)}>
      <div className="h-[400px] w-full">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          fitViewOptions={fitViewOptions}
          nodeTypes={nodeTypes}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={12} size={1} className="bg-muted/30" />
          <Controls className="rounded-xl border border-border !shadow" />
          <MiniMap
            nodeColor={(n) => (highlighted.has(n.id) ? "hsl(var(--primary))" : "hsl(var(--muted))")}
            className="rounded-xl border border-border !bg-background"
          />
          <Panel position="top-left" className="m-2">
            <div className="rounded-lg bg-background/90 px-2 py-1.5 text-xs shadow">
              <span className="font-medium">Node types: </span>
              {Object.entries(NODE_COLORS).slice(0, 5).map(([type, color]) => (
                <span key={type} className="ml-1 inline-flex items-center gap-1">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ background: color }}
                  />
                  {type}
                </span>
              ))}
            </div>
          </Panel>
          {onPlayPath && (
            <Panel position="top-right" className="m-2">
              <button
                onClick={onPlayPath}
                className="rounded-xl bg-primary px-3 py-2 text-xs font-medium text-primary-foreground shadow hover:bg-primary/90"
              >
                Play evidence path
              </button>
            </Panel>
          )}
        </ReactFlow>
      </div>
    </div>
  );
}
