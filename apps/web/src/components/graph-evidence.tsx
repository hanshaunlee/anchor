"use client";

import { useMemo, useState, useCallback, memo, useRef, useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeProps,
  Panel,
  MarkerType,
  Handle,
  Position,
} from "@xyflow/react";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";

const NODE_COLORS: Record<string, string> = {
  person: "hsl(4 70% 52%)",
  entity: "hsl(173 58% 39%)",
  device: "hsl(221 83% 53%)",
  utterance: "hsl(38 92% 50%)",
  intent: "hsl(280 67% 52%)",
  topic: "hsl(24 95% 53%)",
  default: "hsl(215 16% 47%)",
};

const MIN_NODE_R = 8;
const MAX_NODE_R = 28;
const MIN_NODE_SIZE_LABELED = 52;
const MAX_NODE_SIZE_LABELED = 128;
const MAX_LABEL_LEN = 20;
const NODE_TYPE_LABEL = "circle";

/** Compute degree (in + out) per node for sizing. */
function getDegreeMap(edges: SubgraphEdge[]): Map<string, number> {
  const deg = new Map<string, number>();
  for (const e of edges) {
    deg.set(e.src, (deg.get(e.src) ?? 0) + 1);
    deg.set(e.dst, (deg.get(e.dst) ?? 0) + 1);
  }
  return deg;
}

/** Force-directed layout: spread nodes across the canvas (repulsion + edge attraction). */
function forceLayout(
  nodes: SubgraphNode[],
  edges: SubgraphEdge[],
  width: number,
  height: number,
  iterations: number = 80
): Record<string, { x: number; y: number }> {
  const count = nodes.length;
  if (count === 0) return {};
  const centerX = width / 2;
  const centerY = height / 2;
  const area = width * height;
  const k = Math.sqrt(area / Math.max(1, count));
  const isSmallGraph = count <= 15;
  const repel = k * (isSmallGraph ? 1.4 : 0.8);
  const attract = k * (isSmallGraph ? 0.03 : 0.06);
  const positions = new Map<string, { x: number; y: number }>();
  const spread = isSmallGraph ? 0.92 : 0.7;
  if (isSmallGraph && count > 0) {
    const radiusX = (width * spread) / 2;
    const radiusY = (height * spread) / 2;
    nodes.forEach((n, i) => {
      const angle = (i / count) * 2 * Math.PI - Math.PI / 2;
      positions.set(n.id, {
        x: centerX + radiusX * Math.cos(angle),
        y: centerY + radiusY * Math.sin(angle),
      });
    });
  } else {
    const hash = (s: string) => s.split("").reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0);
    nodes.forEach((n) => {
      const u = (Math.abs(hash(n.id)) % 1000) / 1000;
      const v = (Math.abs(hash(n.id + "y")) % 1000) / 1000;
      positions.set(n.id, {
        x: centerX + (u - 0.5) * width * spread,
        y: centerY + (v - 0.5) * height * spread,
      });
    });
  }
  for (let iter = 0; iter < iterations; iter++) {
    const dx = new Map<string, number>();
    const dy = new Map<string, number>();
    nodes.forEach((n) => {
      dx.set(n.id, 0);
      dy.set(n.id, 0);
    });
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      const posA = positions.get(a.id)!;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const posB = positions.get(b.id)!;
        const distX = posB.x - posA.x;
        const distY = posB.y - posA.y;
        const dist = Math.hypot(distX, distY) || 0.001;
        const force = (repel * repel) / dist;
        const fx = (force * distX) / dist;
        const fy = (force * distY) / dist;
        dx.set(a.id, dx.get(a.id)! - fx);
        dy.set(a.id, dy.get(a.id)! - fy);
        dx.set(b.id, dx.get(b.id)! + fx);
        dy.set(b.id, dy.get(b.id)! + fy);
      }
    }
    for (const e of edges) {
      const posA = positions.get(e.src);
      const posB = positions.get(e.dst);
      if (!posA || !posB) continue;
      const distX = posB.x - posA.x;
      const distY = posB.y - posA.y;
      const dist = Math.hypot(distX, distY) || 0.001;
      const force = dist * attract;
      const fx = (force * distX) / dist;
      const fy = (force * distY) / dist;
      dx.set(e.src, dx.get(e.src)! + fx);
      dy.set(e.src, dy.get(e.src)! + fy);
      dx.set(e.dst, dx.get(e.dst)! - fx);
      dy.set(e.dst, dy.get(e.dst)! - fy);
    }
    nodes.forEach((n) => {
      const pos = positions.get(n.id)!;
      const vx = Math.max(-k * 2, Math.min(k * 2, dx.get(n.id)!));
      const vy = Math.max(-k * 2, Math.min(k * 2, dy.get(n.id)!));
      let x = pos.x + vx;
      let y = pos.y + vy;
      const margin = 40;
      x = Math.max(margin, Math.min(width - margin, x));
      y = Math.max(margin, Math.min(height - margin, y));
      positions.set(n.id, { x, y });
    });
  }
  const out: Record<string, { x: number; y: number }> = {};
  positions.forEach((v, id) => { out[id] = v; });
  return out;
}

const LAYOUT_WIDTH = 900;
const LAYOUT_HEIGHT = 480;
const COMPACT_LAYOUT_WIDTH = 380;
const COMPACT_LAYOUT_HEIGHT = 200;

/** Circle node: Handles so edges connect; no label on canvas (tooltip on hover, panel on click). */
const CircleNode = memo(function CircleNode({ data, selected }: NodeProps) {
  const nodeType = (data.nodeType as string) ?? "default";
  const color = NODE_COLORS[nodeType in NODE_COLORS ? nodeType : "default"] ?? NODE_COLORS.default;
  const diameter = typeof data.diameter === "number" ? data.diameter : 24;
  return (
    <>
      <Handle type="target" position={Position.Left} style={{ left: 0, top: "50%", transform: "translate(-50%, -50%)" }} />
      <Handle type="source" position={Position.Right} style={{ right: 0, left: "auto", top: "50%", transform: "translate(50%, -50%)" }} />
      <div
        className={cn(
          "rounded-full border-2 border-white/40 shrink-0 cursor-pointer",
          selected && "ring-2 ring-primary ring-offset-2 ring-offset-background"
        )}
        style={{
          width: diameter,
          height: diameter,
          minWidth: diameter,
          minHeight: diameter,
          background: color,
          boxShadow: "0 1px 3px rgba(0,0,0,0.15)",
          willChange: "transform",
        }}
      />
    </>
  );
});

function toFlowNodes(
  nodes: SubgraphNode[],
  edges: SubgraphEdge[],
  variant: "replay" | "graph",
  highlightIds?: string[],
  opts?: {
    compact?: boolean;
    nodeScaling?: "degree" | "risk" | "static";
    positions?: Record<string, { x: number; y: number }>;
    width?: number;
    height?: number;
  }
): Node[] {
  const compact = opts?.compact ?? false;
  const nodeScaling = opts?.nodeScaling ?? "degree";
  const width = opts?.width ?? (compact ? COMPACT_LAYOUT_WIDTH : LAYOUT_WIDTH);
  const height = opts?.height ?? (compact ? COMPACT_LAYOUT_HEIGHT : LAYOUT_HEIGHT);
  const degreeMap = getDegreeMap(edges);
  const maxDegree = Math.max(1, ...Array.from(degreeMap.values()));
  const highlightSet = highlightIds ? new Set(highlightIds) : new Set<string>();
  const positions = opts?.positions ?? forceLayout(nodes, edges, width, height);
  return nodes.map((n) => {
    const degree = degreeMap.get(n.id) ?? 0;
    const scoreNorm = typeof n.score === "number" ? Math.max(0, Math.min(1, n.score)) : 0;
    const t =
      nodeScaling === "static"
        ? 0.5
        : nodeScaling === "risk"
          ? scoreNorm
          : maxDegree > 0
            ? degree / maxDegree
            : 0;
    const rawLabel = n.label || n.id;
    const shortLabel =
      rawLabel.length > MAX_LABEL_LEN ? rawLabel.slice(0, MAX_LABEL_LEN - 1) + "…" : rawLabel;
    const nodeType = n.type in NODE_COLORS ? n.type : "default";
    const pos = positions[n.id] ?? { x: width / 2, y: height / 2 };

    if (variant === "replay") {
      const minSize = compact ? 28 : MIN_NODE_SIZE_LABELED;
      const maxSize = compact ? 48 : MAX_NODE_SIZE_LABELED;
      const size = Math.round(minSize + t * (maxSize - minSize));
      return {
        id: n.id,
        type: "default",
        position: pos,
        data: { label: shortLabel, score: n.score, nodeType: n.type },
        style: {
          background: NODE_COLORS[nodeType],
          color: "white",
          border: compact ? "1px solid rgba(255,255,255,0.5)" : "2px solid rgba(255,255,255,0.4)",
          borderRadius: compact ? 8 : 14,
          padding: compact ? "4px 8px" : "10px 14px",
          fontSize: compact ? 10 : 12,
          minWidth: size,
          minHeight: size,
          width: size,
          height: size,
          boxShadow: "0 1px 3px rgba(0,0,0,0.12)",
        },
      };
    }

    const diameter = Math.round(MIN_NODE_R * 2 + t * (MAX_NODE_R * 2 - MIN_NODE_R * 2));
    return {
      id: n.id,
      type: NODE_TYPE_LABEL,
      position: pos,
      data: {
        label: rawLabel,
        score: n.score,
        nodeType: n.type,
        diameter,
        degree,
        highlighted: highlightSet.has(n.id),
      },
      style: {
        width: diameter,
        height: diameter,
        boxShadow: highlightSet.has(n.id) ? "0 0 0 3px hsl(var(--primary) / 0.6)" : undefined,
      },
    };
  });
}

/** Explicit edge color so lines are always visible (no reliance on theme vars). */
const GRAPH_EDGE_STROKE = "#475569";
const REPLAY_EDGE_STROKE = "rgba(71, 85, 105, 0.7)";
/** Darker, solid stroke for compact snapshot so edges are clearly visible when scaled down. */
const COMPACT_EDGE_STROKE = "#334155";

function toFlowEdges(
  edges: SubgraphEdge[],
  variant: "replay" | "graph",
  opts?: { compact?: boolean; edgeType?: "smoothstep" | "straight" | "default"; weighted?: boolean }
): Edge[] {
  const compact = opts?.compact ?? false;
  const edgeType = opts?.edgeType ?? "smoothstep";
  const weighted = opts?.weighted ?? true;
  const stroke = compact ? COMPACT_EDGE_STROKE : variant === "graph" ? GRAPH_EDGE_STROKE : REPLAY_EDGE_STROKE;
  const baseWidth = compact ? 3 : variant === "graph" ? 1 : 1;
  return edges.map((e) => ({
    id: `${e.src}-${e.dst}`,
    source: e.src,
    target: e.dst,
    type: edgeType === "straight" || edgeType === "default" ? "default" : "smoothstep",
    animated: false,
    style: {
      strokeWidth: weighted ? Math.max(baseWidth, (e.weight ?? 0.5) * (compact ? 2.5 : 1.2)) : baseWidth,
      stroke,
    },
    markerEnd: MarkerType.ArrowClosed,
  }));
}

export function GraphEvidence({
  nodes,
  edges,
  highlightIds,
  onPlayPath,
  className,
  variant = "graph",
  backgroundVariant = "grid",
  compact = false,
  layoutMode = "force",
  nodeScaling = "degree",
  edgeType = "smoothstep",
  edgeWeighted = true,
}: {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  highlightIds?: string[];
  onPlayPath?: () => void;
  className?: string;
  variant?: "replay" | "graph";
  backgroundVariant?: "grid" | "scientific" | "none";
  /** Smaller layout and thicker edges for alert-detail snapshot */
  compact?: boolean;
  layoutMode?: "force" | "radial";
  nodeScaling?: "degree" | "risk" | "static";
  edgeType?: "smoothstep" | "straight" | "default";
  edgeWeighted?: boolean;
}) {
  const layoutWidth = compact ? COMPACT_LAYOUT_WIDTH : LAYOUT_WIDTH;
  const layoutHeight = compact ? COMPACT_LAYOUT_HEIGHT : LAYOUT_HEIGHT;
  const positionsMemo = useMemo(() => {
    if (nodes.length === 0) return {};
    const centerX = layoutWidth / 2;
    const centerY = layoutHeight / 2;
    const n = nodes.length;
    if (compact) {
      const out: Record<string, { x: number; y: number }> = {};
      const pad = 24;
      const radiusX = (layoutWidth / 2 - pad) * 0.95;
      const radiusY = (layoutHeight / 2 - pad) * 0.95;
      nodes.forEach((node, i) => {
        const angle = (i / Math.max(1, n)) * 2 * Math.PI - Math.PI / 2;
        out[node.id] = {
          x: centerX + radiusX * Math.cos(angle),
          y: centerY + radiusY * Math.sin(angle),
        };
      });
      return out;
    }
    if (layoutMode === "radial") {
      const degreeMap = getDegreeMap(edges);
      const byDegree = [...nodes].sort((a, b) => (degreeMap.get(b.id) ?? 0) - (degreeMap.get(a.id) ?? 0));
      const out: Record<string, { x: number; y: number }> = {};
      const radiusX = layoutWidth * 0.4;
      const radiusY = layoutHeight * 0.4;
      byDegree.forEach((node, i) => {
        const angle = (i / Math.max(1, n)) * 2 * Math.PI - Math.PI / 2;
        out[node.id] = { x: centerX + radiusX * Math.cos(angle), y: centerY + radiusY * Math.sin(angle) };
      });
      return out;
    }
    return forceLayout(nodes, edges, layoutWidth, layoutHeight);
  }, [nodes, edges, layoutMode, layoutWidth, layoutHeight, compact]);
  const initialNodes = useMemo(
    () =>
      toFlowNodes(nodes, edges, variant, highlightIds, {
        compact,
        nodeScaling: variant === "graph" ? nodeScaling : "degree",
        positions: positionsMemo,
        width: layoutWidth,
        height: layoutHeight,
      }),
    [nodes, edges, variant, highlightIds, compact, nodeScaling, positionsMemo, layoutWidth, layoutHeight]
  );
  const initialEdges = useMemo(
    () => toFlowEdges(edges, variant, { compact, edgeType, weighted: edgeWeighted }),
    [edges, variant, compact, edgeType, edgeWeighted]
  );
  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState(initialNodes);
  const [flowEdges, setFlowEdges, onEdgesChange] = useEdgesState(initialEdges);
  useEffect(() => {
    setFlowNodes(initialNodes);
    setFlowEdges(initialEdges);
  }, [initialNodes, initialEdges, setFlowNodes, setFlowEdges]);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [hovered, setHovered] = useState<{ id: string; label: string; nodeType: string; degree?: number; score?: number } | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const TOOLTIP_DELAY_MS = 80;

  const onNodeMouseEnter = useCallback(
    (ev: React.MouseEvent, node: Node) => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = setTimeout(() => {
        hoverTimeoutRef.current = null;
        setTooltipPos({ x: ev.clientX, y: ev.clientY });
        setHovered({
          id: node.id,
          label: String(node.data?.label ?? node.id),
          nodeType: String(node.data?.nodeType ?? "—"),
          degree: typeof node.data?.degree === "number" ? node.data.degree : undefined,
          score: typeof node.data?.score === "number" ? node.data.score : undefined,
        });
      }, TOOLTIP_DELAY_MS);
    },
    []
  );
  const onNodeMouseLeave = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setHovered(null);
  }, []);
  const onNodeMouseMove = useCallback((ev: React.MouseEvent) => {
    if (hovered) setTooltipPos({ x: ev.clientX, y: ev.clientY });
  }, [hovered]);

  const highlighted = useMemo(() => new Set(highlightIds ?? []), [highlightIds]);
  const selectedNode = selectedId ? flowNodes.find((n) => n.id === selectedId) : null;
  const isReplay = variant === "replay";

  const nodeTypes = useMemo(
    () => (isReplay ? undefined : { [NODE_TYPE_LABEL]: CircleNode }),
    [isReplay]
  );
  const fitViewOptions = useMemo(() => ({ padding: 0.2, maxZoom: 2, minZoom: 0.15 }), []);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const next = selectedId === node.id ? null : node.id;
      setSelectedId(next);
      setFlowNodes((nds) =>
        nds.map((n) => ({ ...n, selected: n.id === node.id ? next === node.id : false }))
      );
    },
    [selectedId, setFlowNodes]
  );

  const onPaneClick = useCallback(() => {
    setSelectedId(null);
    setFlowNodes((nds) => nds.map((n) => ({ ...n, selected: false })));
  }, [setFlowNodes]);

  return (
    <div className={cn("rounded-2xl border border-border bg-card overflow-hidden", className)}>
      <div className={cn("w-full", compact ? "h-[220px]" : "h-[520px]")}>
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={isReplay ? undefined : onNodeClick}
          onNodeMouseEnter={isReplay ? undefined : onNodeMouseEnter}
          onNodeMouseLeave={isReplay ? undefined : onNodeMouseLeave}
          onNodeMouseMove={isReplay ? undefined : onNodeMouseMove}
          onPaneClick={isReplay ? undefined : onPaneClick}
          fitView
          fitViewOptions={fitViewOptions}
          nodeTypes={nodeTypes}
          proOptions={{ hideAttribution: true }}
          nodesDraggable={true}
          nodesConnectable={false}
          elementsSelectable={!isReplay}
          selectNodesOnDrag={false}
        >
          {backgroundVariant === "scientific" && (
            <Panel position="top-left" style={{ width: "100%", height: "100%", pointerEvents: "none", margin: 0 }}>
              <div
                className="absolute inset-0 opacity-[0.35]"
                style={{
                  background: "radial-gradient(ellipse 80% 80% at 50% 50%, hsl(var(--muted)) 0%, transparent 70%)",
                }}
              />
              <div
                className="absolute inset-0 opacity-30"
                style={{
                  backgroundImage: "linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)",
                  backgroundSize: "24px 24px",
                }}
              />
            </Panel>
          )}
          {backgroundVariant === "grid" && <Background gap={16} size={0.8} className="bg-muted/20" />}
          {!compact && (
            <>
              <Controls className="rounded-lg border border-border !shadow-sm" showInteractive={false} />
              <MiniMap
                nodeColor={(n) => (highlighted.has(n.id) ? "hsl(var(--primary))" : "hsl(var(--muted))")}
                className="rounded-lg border border-border !bg-background"
              />
            </>
          )}
          {!isReplay && (
            <Panel position="top-left" className="m-2">
              <div className="rounded-lg bg-background/95 px-2.5 py-1.5 text-[11px] shadow border border-border/50">
                <span className="font-medium text-foreground">Types</span>
                <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5">
                  {Object.entries(NODE_COLORS).map(([type, color]) => (
                    <span key={type} className="inline-flex items-center gap-1">
                      <span
                        className="inline-block h-2 w-2 rounded-full shrink-0"
                        style={{ background: color }}
                      />
                      <span className="text-muted-foreground">{type}</span>
                    </span>
                  ))}
                </div>
                <p className="text-muted-foreground mt-1">Hover: tooltip · Click: details · Drag to move</p>
              </div>
            </Panel>
          )}
          {!isReplay && hovered && (
            <div
              className="fixed z-[9999] pointer-events-none rounded-lg px-2.5 py-2 text-[11px] bg-foreground text-background shadow-lg max-w-[220px]"
              style={{ left: tooltipPos.x + 10, top: tooltipPos.y + 8 }}
            >
              <div className="font-medium truncate">{hovered.label}</div>
              <div className="text-background/80 mt-0.5">Type: {hovered.nodeType}</div>
              {hovered.degree != null && <div className="text-background/80 mt-0.5">Degree: {hovered.degree}</div>}
              {hovered.score != null && <div className="text-background/80 mt-0.5">Risk contribution: {(hovered.score * 100).toFixed(0)}%</div>}
            </div>
          )}
          {!isReplay && selectedNode && selectedNode.data && (
            <Panel position="top-right" className="m-2 max-w-[200px]">
              <div className="rounded-md bg-foreground/90 text-background px-2 py-1.5 text-[11px] shadow-sm">
                <span className="font-medium truncate block" title={String(selectedNode.data.label)}>
                  {String(selectedNode.data.label)}
                </span>
                <span className="text-background/80">{String(selectedNode.data.nodeType ?? "—")}</span>
              </div>
            </Panel>
          )}
          {onPlayPath && (
            <Panel position="bottom-right" className="m-2">
              <button
                onClick={onPlayPath}
                className="rounded-lg bg-primary px-2.5 py-1.5 text-xs font-medium text-primary-foreground shadow hover:bg-primary/90"
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
