import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";

/** Compute degree (in + out) per node. */
export function getDegreeMap(edges: SubgraphEdge[]): Map<string, number> {
  const deg = new Map<string, number>();
  for (const e of edges) {
    deg.set(e.src, (deg.get(e.src) ?? 0) + 1);
    deg.set(e.dst, (deg.get(e.dst) ?? 0) + 1);
  }
  return deg;
}

/** Connected components via union-find. */
function connectedComponents(nodes: SubgraphNode[], edges: SubgraphEdge[]): Map<string, string> {
  const parent = new Map<string, string>();
  for (const n of nodes) parent.set(n.id, n.id);
  const find = (id: string): string => {
    const p = parent.get(id)!;
    return p === id ? id : (parent.set(id, find(p)), parent.get(id)!);
  };
  const union = (a: string, b: string) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };
  for (const e of edges) union(e.src, e.dst);
  return parent;
}

/** Component sizes: root id -> size. */
function componentSizes(parent: Map<string, string>): Map<string, number> {
  const roots = new Map<string, string>();
  for (const [id] of parent) {
    let r = id;
    while (parent.get(r) !== r) r = parent.get(r)!;
    roots.set(id, r);
  }
  const sizes = new Map<string, number>();
  for (const r of roots.values()) sizes.set(r, (sizes.get(r) ?? 0) + 1);
  return sizes;
}

/** Local clustering coefficient for one node: 2 * triangles / (degree * (degree - 1)). */
function clusteringForNode(
  nodeId: string,
  degreeMap: Map<string, number>,
  neighbors: Map<string, Set<string>>
): number {
  const deg = degreeMap.get(nodeId) ?? 0;
  if (deg < 2) return 0;
  const neigh = neighbors.get(nodeId);
  if (!neigh) return 0;
  let triangles = 0;
  for (const u of neigh) {
    for (const v of neigh) {
      if (u >= v) continue;
      if (neighbors.get(u)?.has(v)) triangles++;
    }
  }
  return (2 * triangles) / (deg * (deg - 1));
}

/** Build neighbor sets from edges. */
function neighborSets(edges: SubgraphEdge[]): Map<string, Set<string>> {
  const m = new Map<string, Set<string>>();
  for (const e of edges) {
    if (!m.has(e.src)) m.set(e.src, new Set());
    m.get(e.src)!.add(e.dst);
    if (!m.has(e.dst)) m.set(e.dst, new Set());
    m.get(e.dst)!.add(e.src);
  }
  return m;
}

export type GraphStructuralMetrics = {
  nodeCount: number;
  edgeCount: number;
  density: number;
  connectedComponents: number;
  largestComponentSize: number;
  avgClusteringCoefficient: number;
  topCentralNodeIds: string[];
};

export function computeGraphMetrics(
  nodes: SubgraphNode[],
  edges: SubgraphEdge[],
  topK: number = 5
): GraphStructuralMetrics {
  const nodeCount = nodes.length;
  const edgeCount = edges.length;
  const maxPossibleEdges = nodeCount <= 1 ? 0 : (nodeCount * (nodeCount - 1)) / 2;
  const density = maxPossibleEdges > 0 ? edgeCount / maxPossibleEdges : 0;

  const parent = connectedComponents(nodes, edges);
  const sizes = componentSizes(parent);
  const componentCount = sizes.size;
  const largestComponentSize = sizes.size > 0 ? Math.max(...sizes.values()) : 0;

  const degreeMap = getDegreeMap(edges);
  const neighbors = neighborSets(edges);
  let sumClustering = 0;
  let countWithDegree2 = 0;
  for (const n of nodes) {
    const c = clusteringForNode(n.id, degreeMap, neighbors);
    if (degreeMap.get(n.id) && degreeMap.get(n.id)! >= 2) {
      sumClustering += c;
      countWithDegree2++;
    }
  }
  const avgClusteringCoefficient = countWithDegree2 > 0 ? sumClustering / countWithDegree2 : 0;

  const byDegree = [...nodes].sort((a, b) => (degreeMap.get(b.id) ?? 0) - (degreeMap.get(a.id) ?? 0));
  const topCentralNodeIds = byDegree.slice(0, topK).map((n) => n.id);

  return {
    nodeCount,
    edgeCount,
    density,
    connectedComponents: componentCount,
    largestComponentSize,
    avgClusteringCoefficient: Math.round(avgClusteringCoefficient * 100) / 100,
    topCentralNodeIds,
  };
}
