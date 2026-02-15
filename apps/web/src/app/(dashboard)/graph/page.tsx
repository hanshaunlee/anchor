"use client";

import { useMemo, useState, useCallback, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  useGraphEvidence,
  useGraphNeo4jStatus,
  useSyncGraphToNeo4jMutation,
  useRiskSignals,
} from "@/hooks/use-api";
import { TopologyHeader } from "@/components/graph/TopologyHeader";
import { TopologyControls } from "@/components/graph/TopologyControls";
import type { LayoutMode, NodeScaling, EdgeRenderMode } from "@/components/graph/TopologyControls";
import { ScientificGraphCanvas } from "@/components/graph/ScientificGraphCanvas";
import { GraphAnalysisPanel } from "@/components/graph/GraphAnalysisPanel";
import { GraphStatsFooter } from "@/components/graph/GraphStatsFooter";
import { computeGraphMetrics } from "@/lib/graph-metrics";
import { Skeleton } from "@/components/ui/skeleton";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";
import { api } from "@/lib/api";
import { Copy } from "lucide-react";

const MAX_DISPLAY_NODES = 100;

function capByDegree(
  nodes: SubgraphNode[],
  edges: SubgraphEdge[],
  maxNodes: number
): { nodes: SubgraphNode[]; edges: SubgraphEdge[]; capped: boolean; totalNodes: number } {
  if (nodes.length <= maxNodes)
    return { nodes, edges, capped: false, totalNodes: nodes.length };
  const degree = new Map<string, number>();
  for (const e of edges) {
    degree.set(e.src, (degree.get(e.src) ?? 0) + 1);
    degree.set(e.dst, (degree.get(e.dst) ?? 0) + 1);
  }
  const byDegree = [...nodes].sort((a, b) => (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0));
  const keep = new Set(byDegree.slice(0, maxNodes).map((n) => n.id));
  const filteredEdges = edges.filter((e) => keep.has(e.src) && keep.has(e.dst));
  return {
    nodes: byDegree.slice(0, maxNodes),
    edges: filteredEdges,
    capped: true,
    totalNodes: nodes.length,
  };
}

function buildNeo4jBrowserUrl(browserUrl: string | null, connectUrl: string | null): string | null {
  if (!browserUrl) return null;
  const base = (browserUrl ?? "http://localhost:7474").replace(/\/?$/, "");
  const params = new URLSearchParams();
  if (connectUrl) params.set("connectURL", connectUrl);
  params.set("cmd", "edit");
  params.set("arg", "MATCH (e:Entity) RETURN e LIMIT 50");
  return `${base}/browser?${params.toString()}`;
}

export default function GraphViewPage() {
  const searchParams = useSearchParams();
  const [liveSync, setLiveSync] = useState(false);
  const [resetViewKey, setResetViewKey] = useState(0);
  const [highlightNodeIds, setHighlightNodeIds] = useState<string[]>([]);
  const [passwordCopied, setPasswordCopied] = useState(false);

  const [onlySuspicious, setOnlySuspicious] = useState(false);
  const [onlyRecent7d, setOnlyRecent7d] = useState(false);
  const [highlightRings, setHighlightRings] = useState(false);
  const [highlightMotifs, setHighlightMotifs] = useState(false);
  const [highlightAlertNodes, setHighlightAlertNodes] = useState(false);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("force");
  const [nodeScaling, setNodeScaling] = useState<NodeScaling>("degree");
  const [edgeRenderMode, setEdgeRenderMode] = useState<EdgeRenderMode>("weighted");

  const { data: graphData, isLoading } = useGraphEvidence({ liveSync });
  const { data: neo4jStatus } = useGraphNeo4jStatus();
  const syncMutation = useSyncGraphToNeo4jMutation();
  const { data: signalsData } = useRiskSignals({ status: "open", limit: 20 });
  const activeAlerts = useMemo(() => signalsData?.signals ?? [], [signalsData?.signals]);

  const { nodes: cappedNodes, edges: cappedEdges, capped, totalNodes } = useMemo(() => {
    const rawNodes = graphData?.nodes ?? [];
    const rawEdges = graphData?.edges ?? [];
    return capByDegree(rawNodes, rawEdges, MAX_DISPLAY_NODES);
  }, [graphData]);

  const { nodes, edges } = useMemo(() => {
    if (!highlightAlertNodes || highlightNodeIds.length === 0)
      return { nodes: cappedNodes, edges: cappedEdges };
    const idSet = new Set(highlightNodeIds);
    const nodes = cappedNodes.filter((n) => idSet.has(n.id));
    const edges = cappedEdges.filter((e) => idSet.has(e.src) && idSet.has(e.dst));
    return { nodes, edges };
  }, [cappedNodes, cappedEdges, highlightAlertNodes, highlightNodeIds]);

  const hasData = nodes.length > 0 || edges.length > 0;
  const neo4jEnabled = neo4jStatus?.enabled ?? false;
  const browserUrl = neo4jStatus?.browser_url ?? null;
  const connectUrl = neo4jStatus?.connect_url ?? null;
  const neo4jPassword = neo4jStatus?.password ?? null;
  const neo4jBrowserUrl = useMemo(() => buildNeo4jBrowserUrl(browserUrl, connectUrl), [browserUrl, connectUrl]);

  const metrics = useMemo(() => (hasData ? computeGraphMetrics(nodes, edges) : null), [nodes, edges, hasData]);
  const nodeLabels = useMemo(() => {
    const m = new Map<string, string>();
    nodes.forEach((n) => m.set(n.id, n.label ?? n.id));
    return m;
  }, [nodes]);

  const handleExportSubgraph = useCallback(() => {
    const payload = { nodes, edges, exportedAt: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `topology-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges]);

  const handleResetView = useCallback(() => {
    setResetViewKey((k) => k + 1);
  }, []);

  const handleHighlightAlert = useCallback(async (signalId: string) => {
    try {
      const detail = await api.getRiskSignal(signalId);
      const ids = detail?.entity_ids ?? [];
      setHighlightNodeIds(ids);
    } catch {
      setHighlightNodeIds([]);
    }
  }, []);

  const highlightFromUrl = searchParams.get("highlight");
  const highlightRingId = searchParams.get("highlightRing");

  useEffect(() => {
    if (highlightFromUrl) {
      handleHighlightAlert(highlightFromUrl);
      setHighlightAlertNodes(true);
    }
  }, [highlightFromUrl, handleHighlightAlert]);

  useEffect(() => {
    if (!highlightRingId) return;
    let cancelled = false;
    (async () => {
      try {
        const ring = await api.getProtectionRing(highlightRingId) as { members?: Array<{ entity_id: string | null }> };
        if (cancelled || !ring?.members) return;
        const ids = ring.members.map((m) => m.entity_id).filter((id): id is string => id != null);
        setHighlightNodeIds(ids);
        setHighlightAlertNodes(true);
      } catch {
        try {
          const legacy = await api.getRing(highlightRingId) as { members?: Array<{ entity_id: string | null }> };
          if (cancelled || !legacy?.members) return;
          const ids = legacy.members.map((m) => m.entity_id).filter((id): id is string => id != null);
          setHighlightNodeIds(ids);
          setHighlightAlertNodes(true);
        } catch {
          // ignore
        }
      }
    })();
    return () => { cancelled = true; };
  }, [highlightRingId]);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-[400px] w-full rounded-2xl" />
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-[calc(100vh-8rem)]">
      <div className="space-y-4 flex-shrink-0">
        <TopologyHeader
          liveSync={liveSync}
          onLiveSyncChange={setLiveSync}
          onSyncNeo4j={() => syncMutation.mutate()}
          syncNeo4jPending={syncMutation.isPending}
          neo4jEnabled={neo4jEnabled}
          onExportSubgraph={handleExportSubgraph}
          onResetView={handleResetView}
          neo4jBrowserUrl={neo4jBrowserUrl}
          hasData={hasData}
        />

        <TopologyControls
          onlySuspicious={onlySuspicious}
          onOnlySuspiciousChange={setOnlySuspicious}
          onlyRecent7d={onlyRecent7d}
          onOnlyRecent7dChange={setOnlyRecent7d}
          highlightRings={highlightRings}
          onHighlightRingsChange={setHighlightRings}
          highlightMotifs={highlightMotifs}
          onHighlightMotifsChange={setHighlightMotifs}
          highlightAlertNodes={highlightAlertNodes}
          onHighlightAlertNodesChange={setHighlightAlertNodes}
          layoutMode={layoutMode}
          onLayoutModeChange={setLayoutMode}
          nodeScaling={nodeScaling}
          onNodeScalingChange={setNodeScaling}
          edgeRenderMode={edgeRenderMode}
          onEdgeRenderModeChange={setEdgeRenderMode}
        />
      </div>

      <div className="flex gap-4 flex-1 min-h-0 mt-4">
        <div className="flex-1 min-w-0">
          {hasData ? (
            <ScientificGraphCanvas
              nodes={nodes}
              edges={edges}
              highlightNodeIds={highlightNodeIds.length > 0 ? highlightNodeIds : undefined}
              resetViewKey={resetViewKey}
              pulseOnUpdate
              layoutMode={layoutMode}
              nodeScaling={nodeScaling}
              edgeRenderMode={edgeRenderMode}
            />
          ) : (
            <div className="flex h-[520px] items-center justify-center rounded-2xl border border-dashed border-border bg-muted/30 text-muted-foreground text-sm">
              No graph data. Run the pipeline or ingest events for this household.
            </div>
          )}
        </div>
        <div className="w-72 shrink-0">
          <GraphAnalysisPanel
            metrics={metrics}
            nodeLabels={nodeLabels}
            activeAlerts={activeAlerts}
            onHighlightAlert={handleHighlightAlert}
            defaultOpen={false}
          />
        </div>
      </div>

      {hasData && metrics && (
        <GraphStatsFooter
          nodeCount={metrics.nodeCount}
          edgeCount={metrics.edgeCount}
          density={metrics.density}
          connectedComponents={metrics.connectedComponents}
          largestComponentSize={metrics.largestComponentSize}
          ringsCount={undefined}
          watchlistedCount={undefined}
          driftStatus="Stable"
          className="mt-4"
        />
      )}

      {neo4jEnabled && browserUrl && (
        <div className="text-xs text-muted-foreground space-y-1 mt-4 p-4 rounded-xl bg-muted/20">
          <p>
            {connectUrl
              ? "The link opens Neo4j Browser with username and URL pre-filled. Neo4j does not allow pre-filling the password; if a login form appears, use the password below and click Connect."
              : "Neo4j Browser must be running for the link to work. Set NEO4J_PASSWORD in the API .env so the link can pre-fill login."}
          </p>
          {connectUrl && neo4jPassword && (
            <div className="flex flex-wrap items-center gap-2">
              <span>Password:</span>
              <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">{neo4jPassword}</code>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                onClick={() => {
                  void navigator.clipboard.writeText(neo4jPassword).then(() => {
                    setPasswordCopied(true);
                    setTimeout(() => setPasswordCopied(false), 2000);
                  });
                }}
              >
                <Copy className="h-3 w-3 mr-1" />
                {passwordCopied ? "Copied!" : "Copy"}
              </Button>
            </div>
          )}
        </div>
      )}

      {!neo4jEnabled && (
        <Card className="rounded-2xl border-dashed border-border bg-muted/20 mt-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Neo4j (optional)</CardTitle>
            <p className="text-xs text-muted-foreground">
              To sync this graph to Neo4j and use Cypher in Neo4j Browser, set{" "}
              <code className="rounded bg-muted px-1">NEO4J_URI</code> (and optionally{" "}
              <code className="rounded bg-muted px-1">NEO4J_USER</code>,{" "}
              <code className="rounded bg-muted px-1">NEO4J_PASSWORD</code>) in the API .env, then restart the API.
              See <code className="rounded bg-muted px-1">docs/NEO4J_SETUP.md</code>.
            </p>
          </CardHeader>
        </Card>
      )}
    </div>
  );
}
