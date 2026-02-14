"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { GraphEvidence } from "@/components/graph-evidence";
import {
  useGraphEvidence,
  useGraphNeo4jStatus,
  useSyncGraphToNeo4jMutation,
} from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";
import { ExternalLink, Network, RefreshCw } from "lucide-react";

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

export default function GraphViewPage() {
  const { data: graphData, isLoading } = useGraphEvidence();
  const { data: neo4jStatus } = useGraphNeo4jStatus();
  const syncMutation = useSyncGraphToNeo4jMutation();

  const { nodes, edges, capped, totalNodes } = useMemo(() => {
    const rawNodes = graphData?.nodes ?? [];
    const rawEdges = graphData?.edges ?? [];
    return capByDegree(rawNodes, rawEdges, MAX_DISPLAY_NODES);
  }, [graphData]);
  const hasData = nodes.length > 0 || edges.length > 0;
  const neo4jEnabled = neo4jStatus?.enabled ?? false;
  const browserUrl = neo4jStatus?.browser_url ?? null;
  const connectUrl = neo4jStatus?.connect_url ?? null;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-[400px] w-full rounded-2xl" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Network className="h-6 w-6 text-muted-foreground" />
          <h1 className="text-2xl font-semibold tracking-tight">
            Graph view
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {neo4jEnabled && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => syncMutation.mutate()}
                disabled={syncMutation.isPending}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${syncMutation.isPending ? "animate-spin" : ""}`} />
                Sync to Neo4j
              </Button>
              <a
                href={(() => {
                  const base = (browserUrl ?? "http://localhost:7474").replace(/\/?$/, "");
                  const params = new URLSearchParams();
                  if (connectUrl) {
                    params.set("connectURL", connectUrl);
                  }
                  params.set("cmd", "edit");
                  params.set("arg", "MATCH (e:Entity) RETURN e LIMIT 50");
                  return `${base}/browser?${params.toString()}`;
                })()}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium border border-input bg-background px-4 py-2 ring-offset-background hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none"
              >
                <ExternalLink className="h-4 w-4 mr-2" />
                Open in Neo4j Browser
              </a>
            </>
          )}
        </div>
      </div>
      {neo4jEnabled && browserUrl && (
        <p className="text-xs text-muted-foreground">
          {connectUrl
            ? "The link opens Neo4j Browser with connection details pre-filled; if a login form appears, click Connect."
            : "Neo4j Browser must be running for the link to work. Set NEO4J_PASSWORD in the API .env so the link can pre-fill login."}
        </p>
      )}

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            Household evidence graph
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            {hasData
              ? "People, places, and topics extracted from sessions and how they connect. Larger nodes have more connections. Use zoom and pan to explore; sync to Neo4j for the full graph and Cypher queries."
              : "No events yet. Ingest events or run the pipeline to build the household graph."}
          </p>
          {hasData && (
            <p className="text-xs text-muted-foreground">
              {capped
                ? `Showing top ${nodes.length} of ${totalNodes} entities (${edges.length} relationships). Full graph available in Neo4j.`
                : `${nodes.length} entities, ${edges.length} relationships.`}
            </p>
          )}
        </CardHeader>
        <CardContent>
          {hasData ? (
            <GraphEvidence variant="graph" nodes={nodes} edges={edges} className="mt-2" />
          ) : (
            <div className="flex h-[320px] items-center justify-center rounded-2xl border border-dashed border-border bg-muted/30 text-muted-foreground text-sm">
              No graph data. Run the pipeline or ingest events for this household.
            </div>
          )}
        </CardContent>
      </Card>

      {neo4jEnabled && !browserUrl && (
        <p className="text-xs text-muted-foreground">
          Neo4j is configured (remote). Use Neo4j Desktop or your instance URL to open the browser.
        </p>
      )}

      {!neo4jEnabled && (
        <Card className="rounded-2xl border-dashed border-border bg-muted/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Neo4j (optional)</CardTitle>
            <p className="text-xs text-muted-foreground">
              To sync this graph to Neo4j and use Cypher in Neo4j Browser, set <code className="rounded bg-muted px-1">NEO4J_URI</code> (and optionally <code className="rounded bg-muted px-1">NEO4J_USER</code>, <code className="rounded bg-muted px-1">NEO4J_PASSWORD</code>) in the API .env, then restart the API. See <code className="rounded bg-muted px-1">docs/NEO4J_SETUP.md</code>.
            </p>
          </CardHeader>
        </Card>
      )}
    </div>
  );
}
