"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { GraphEvidence } from "@/components/graph-evidence";
import { cn } from "@/lib/utils";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";
import type { EventListItem } from "@/lib/api/schemas";
import { Calendar, Network, Layers, ChevronDown, ChevronUp } from "lucide-react";

const TIMELINE_VISIBLE = 5;
const TIMELINE_EXPAND_EXTRA = 10;

export type EvidenceOverviewProps = {
  /** Key events (session_events) */
  events: EventListItem[];
  /** Subgraph for main view */
  subgraph: { nodes: SubgraphNode[]; edges: SubgraphEdge[] } | null;
  /** Deep-dive subgraph if computed */
  deepDiveSubgraph: { nodes: SubgraphNode[]; edges: SubgraphEdge[] } | null;
  /** "model" | "deep_dive" */
  graphView: "model" | "deep_dive";
  onGraphViewChange: (v: "model" | "deep_dive") => void;
  modelAvailable: boolean;
  onRequestDeepDive: () => void;
  deepDivePending: boolean;
  /** Structural motifs (triadic, star, etc.) */
  motifs: string[];
  /** Semantic tags (urgency, new_contact, etc.) */
  semanticTags?: string[];
  /** Show independence violation badge */
  independenceViolation?: boolean;
  /** When set, "Play evidence path" navigates to graph with this alert highlighted */
  signalId?: string;
};

export function EvidenceOverview({
  events,
  subgraph,
  deepDiveSubgraph,
  graphView,
  onGraphViewChange,
  modelAvailable,
  onRequestDeepDive,
  deepDivePending,
  motifs,
  semanticTags = [],
  independenceViolation,
  signalId,
}: EvidenceOverviewProps) {
  const router = useRouter();
  const [timelineExpanded, setTimelineExpanded] = useState(false);
  const onPlayPath = useCallback(() => {
    if (signalId) router.push(`/graph?highlight=${encodeURIComponent(signalId)}`);
  }, [router, signalId]);
  const visibleEvents = timelineExpanded
    ? events.slice(0, TIMELINE_VISIBLE + TIMELINE_EXPAND_EXTRA)
    : events.slice(0, TIMELINE_VISIBLE);
  const hasMoreEvents = events.length > TIMELINE_VISIBLE;

  const nodes = graphView === "deep_dive" && deepDiveSubgraph?.nodes?.length
    ? deepDiveSubgraph.nodes
    : subgraph?.nodes ?? [];
  const edges = graphView === "deep_dive" && deepDiveSubgraph?.edges?.length
    ? deepDiveSubgraph.edges
    : subgraph?.edges ?? [];
  const hasGraph = nodes.length > 0 || edges.length > 0;

  return (
    <div className="grid gap-6 md:grid-cols-3">
      {/* Timeline */}
      <Card className="rounded-2xl shadow-sm border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Calendar className="h-4 w-4" />
            Timeline
          </CardTitle>
          <p className="text-muted-foreground text-xs">Key events</p>
        </CardHeader>
        <CardContent>
          {events.length === 0 ? (
            <p className="text-muted-foreground text-sm">No events loaded or redacted.</p>
          ) : (
            <ul className="space-y-2">
              {visibleEvents.map((e) => (
                <li
                  key={e.id}
                  className="flex gap-2 rounded-lg border border-border p-2 text-xs"
                >
                  <span className="text-muted-foreground shrink-0">
                    {new Date(e.ts).toLocaleTimeString()}
                  </span>
                  <span className="font-medium truncate">{e.event_type}</span>
                  {e.text_redacted ? (
                    <span className="text-muted-foreground italic truncate">Redacted</span>
                  ) : (
                    <span className="truncate text-muted-foreground">
                      {typeof (e.payload as Record<string, unknown>)?.text === "string"
                        ? ((e.payload as Record<string, unknown>).text as string).slice(0, 40) + "…"
                        : ""}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {hasMoreEvents && (
            <Button
              variant="ghost"
              size="sm"
              className="mt-2 w-full rounded-xl text-xs"
              onClick={() => setTimelineExpanded(!timelineExpanded)}
            >
              {timelineExpanded ? (
                <>Show less <ChevronUp className="h-3 w-3 ml-1 inline" /></>
              ) : (
                <>Show more ({events.length - TIMELINE_VISIBLE} more) <ChevronDown className="h-3 w-3 ml-1 inline" /></>
              )}
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Graph snapshot */}
      <Card className="rounded-2xl shadow-sm border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Network className="h-4 w-4" />
            Graph snapshot
          </CardTitle>
          {modelAvailable && (
            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="button"
                onClick={() => onGraphViewChange("model")}
                className={cn(
                  "rounded-lg px-2 py-1 text-xs font-medium",
                  graphView === "model" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                )}
              >
                PGExplainer
              </button>
              <button
                type="button"
                onClick={() => onGraphViewChange("deep_dive")}
                className={cn(
                  "rounded-lg px-2 py-1 text-xs font-medium",
                  graphView === "deep_dive" ? "bg-primary text-primary-foreground" : "bg-muted hover:bg-muted/80"
                )}
              >
                Deep dive
              </button>
            </div>
          )}
        </CardHeader>
        <CardContent>
          {!hasGraph ? (
            <p className="text-muted-foreground text-sm">No graph evidence for this signal.</p>
          ) : graphView === "deep_dive" && !deepDiveSubgraph?.nodes?.length ? (
            <div className="space-y-2 py-2">
              <p className="text-muted-foreground text-xs">Deep dive not computed yet.</p>
              <Button
                size="sm"
                variant="secondary"
                className="rounded-xl"
                disabled={deepDivePending}
                onClick={onRequestDeepDive}
              >
                {deepDivePending ? "Computing…" : "Compute deep dive"}
              </Button>
            </div>
          ) : (
            <div className="w-full rounded-xl border border-border overflow-hidden bg-muted/20 [&_.react-flow]:!min-h-0">
              <GraphEvidence
                variant="replay"
                nodes={nodes}
                edges={edges}
                onPlayPath={signalId ? onPlayPath : undefined}
                compact
              />
            </div>
          )}
          <div className="flex gap-2 mt-2">
            <Link href="/graph" className="flex-1">
              <Button variant="outline" size="sm" className="w-full rounded-xl">
                Open full graph
              </Button>
            </Link>
            {signalId && (
              <Link href={`/replay?alert=${signalId}`} className="flex-1">
                <Button variant="outline" size="sm" className="w-full rounded-xl">
                  Open in Scenario Replay
                </Button>
              </Link>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Pattern signals */}
      <Card className="rounded-2xl shadow-sm border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Layers className="h-4 w-4" />
            Pattern signals
          </CardTitle>
          <p className="text-muted-foreground text-xs">Motifs and tags</p>
        </CardHeader>
        <CardContent className="space-y-2">
          {motifs.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {motifs.map((m, i) => (
                <Badge key={i} variant="secondary" className="rounded-lg text-xs">
                  {String(m)}
                </Badge>
              ))}
            </div>
          )}
          {semanticTags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {semanticTags.map((t, i) => (
                <Badge key={i} variant="outline" className="rounded-lg text-xs">
                  {String(t)}
                </Badge>
              ))}
            </div>
          )}
          {independenceViolation && (
            <Badge variant="destructive" className="rounded-lg text-xs">
              Independence violation
            </Badge>
          )}
          {motifs.length === 0 && semanticTags.length === 0 && !independenceViolation && (
            <p className="text-muted-foreground text-sm">No structural motifs or semantic tags.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
