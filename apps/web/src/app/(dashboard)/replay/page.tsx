"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
import { ScientificGraphCanvas } from "@/components/graph/ScientificGraphCanvas";
import { TopologyControls } from "@/components/graph/TopologyControls";
import type { LayoutMode, NodeScaling, EdgeRenderMode } from "@/components/graph/TopologyControls";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import { Play, RotateCcw, Download, ExternalLink } from "lucide-react";
import { api } from "@/lib/api";
import { useAppStore } from "@/store/use-app-store";
import {
  useGraphEvidence,
  useAgentsStatus,
  useAgentTrace,
  useRiskSignals,
  useRiskSignalDetail,
} from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";

const MAX_DISPLAY_NODES = 100;

function capByDegree(
  nodes: SubgraphNode[],
  edges: SubgraphEdge[],
  maxNodes: number
): { nodes: SubgraphNode[]; edges: SubgraphEdge[] } {
  if (nodes.length <= maxNodes) return { nodes, edges };
  const degree = new Map<string, number>();
  for (const e of edges) {
    degree.set(e.src, (degree.get(e.src) ?? 0) + 1);
    degree.set(e.dst, (degree.get(e.dst) ?? 0) + 1);
  }
  const byDegree = [...nodes].sort((a, b) => (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0));
  const keep = new Set(byDegree.slice(0, maxNodes).map((n) => n.id));
  const filteredEdges = edges.filter((e) => keep.has(e.src) && keep.has(e.dst));
  return { nodes: byDegree.slice(0, maxNodes), edges: filteredEdges };
}

function stepTraceToTraceSteps(
  stepTrace: Array<{ step?: string; status?: string; notes?: string; error?: string }>
): TraceStep[] {
  if (!Array.isArray(stepTrace)) return [];
  return stepTrace.map((s) => {
    const raw = (s.status ?? "").toLowerCase();
    const status: TraceStep["status"] =
      raw === "ok" || raw === "success" || raw === "completed" ? "success"
      : raw === "error" || raw === "fail" ? "fail"
      : "warn";
    return {
      step: s.step ?? "Step",
      description: s.notes ?? s.error ?? "",
      status,
    };
  });
}

export default function ReplayPage() {
  const searchParams = useSearchParams();
  const alertId = searchParams.get("alert");
  const sourceRedteam = searchParams.get("source") === "redteam";
  const demoMode = useAppStore((s) => s.demoMode);

  const [scenarioSource, setScenarioSource] = useState<"last_run" | "fixture" | "api_demo">("last_run");
  const [playing, setPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [chartRevealIndex, setChartRevealIndex] = useState(0);
  const [loadingFromApi, setLoadingFromApi] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [resetViewKey, setResetViewKey] = useState(0);

  const [layoutMode, setLayoutMode] = useState<LayoutMode>("force");
  const [nodeScaling, setNodeScaling] = useState<NodeScaling>("degree");
  const [edgeRenderMode, setEdgeRenderMode] = useState<EdgeRenderMode>("weighted");
  const [onlySuspicious, setOnlySuspicious] = useState(false);
  const [onlyRecent7d, setOnlyRecent7d] = useState(false);
  const [highlightRings, setHighlightRings] = useState(false);
  const [highlightMotifs, setHighlightMotifs] = useState(false);
  const [highlightAlertNodes, setHighlightAlertNodes] = useState(false);

  const { data: graphData, isLoading: graphLoading } = useGraphEvidence();
  const { data: statusData } = useAgentsStatus();
  const { data: signalsData } = useRiskSignals({ status: "open", limit: 20, severityMin: undefined });
  const supervisorStatus = statusData?.agents?.find((a: { agent_name?: string }) => a.agent_name === "supervisor");
  const supervisorRunId = supervisorStatus?.last_run_id ?? null;
  const { data: supervisorTrace } = useAgentTrace(supervisorRunId, "supervisor");
  const { data: alertDetail } = useRiskSignalDetail(alertId);

  const { nodes: cappedNodes, edges: cappedEdges } = useMemo(() => {
    const rawNodes = graphData?.nodes ?? [];
    const rawEdges = graphData?.edges ?? [];
    return capByDegree(rawNodes, rawEdges, MAX_DISPLAY_NODES);
  }, [graphData]);

  const highlightNodeIds = useMemo(() => {
    if (alertId && alertDetail?.entity_ids?.length) {
      return alertDetail.entity_ids.map((id) => String(id));
    }
    return [];
  }, [alertId, alertDetail?.entity_ids]);

  const { nodes, edges } = useMemo(() => {
    if (!highlightAlertNodes || highlightNodeIds.length === 0)
      return { nodes: cappedNodes, edges: cappedEdges };
    const idSet = new Set(highlightNodeIds);
    return {
      nodes: cappedNodes.filter((n) => idSet.has(n.id)),
      edges: cappedEdges.filter((e) => idSet.has(e.src) && idSet.has(e.dst)),
    };
  }, [cappedNodes, cappedEdges, highlightAlertNodes, highlightNodeIds]);

  const hasGraphData = nodes.length > 0 || edges.length > 0;

  const lastRunTrace = useMemo(() => {
    const trace = supervisorTrace?.step_trace as Array<{ step?: string; status?: string; notes?: string; error?: string }> | undefined;
    return trace?.length ? stepTraceToTraceSteps(trace) : [];
  }, [supervisorTrace?.step_trace]);

  const timelineFromSignals = useMemo(() => {
    const signals = signalsData?.signals ?? [];
    return signals.slice(0, 15).map((s, i) => ({
      t: i * 20,
      score: s.score,
      label: (s.summary ?? s.signal_type ?? `Alert ${i + 1}`).slice(0, 40) + (s.summary && s.summary.length > 40 ? "…" : ""),
    }));
  }, [signalsData?.signals]);

  const traceSteps = useMemo(() => {
    if (alertId && alertDetail) {
      return [
        {
          step: "Alert",
          description: alertDetail.summary ?? alertDetail.explanation?.summary ?? "Risk signal raised.",
          status: "success" as const,
        },
      ];
    }
    if (scenarioSource === "last_run" && lastRunTrace.length > 0) return lastRunTrace;
    if (scenarioSource === "api_demo" && lastRunTrace.length > 0) return lastRunTrace;
    return [];
  }, [alertId, alertDetail, scenarioSource, lastRunTrace]);

  const timeline = useMemo(() => {
    if (alertId && alertDetail) {
      return [{ t: 0, score: alertDetail.score, label: alertDetail.summary?.slice(0, 30) ?? "Alert" }];
    }
    if (timelineFromSignals.length > 0) return timelineFromSignals;
    return [{ t: 0, score: 0.5, label: "No recent signals" }];
  }, [alertId, alertDetail, timelineFromSignals]);

  const title = useMemo(() => {
    if (alertId && alertDetail?.summary) return alertDetail.summary.slice(0, 80);
    if (scenarioSource === "last_run" && supervisorRunId)
      return "Last safety check run";
    return "Scenario Replay";
  }, [alertId, alertDetail?.summary, scenarioSource, supervisorRunId]);

  const description = useMemo(() => {
    if (alertId && alertDetail) return "Evidence and entities involved in this alert. Same graph as Graph view.";
    if (scenarioSource === "last_run")
      return "Your household graph and the last investigation run. Timeline shows recent open alerts.";
    return "Replay a scenario: pick an alert or use the last run. Graph is the same as Graph view.";
  }, [alertId, alertDetail, scenarioSource]);

  const chartData = useMemo(() => {
    const slice = timeline.slice(0, chartRevealIndex + 1).map((p) => ({ ...p, t: String(p.t) }));
    if (slice.length === 0 && timeline.length > 0) return [{ ...timeline[0], t: String(timeline[0].t) }];
    return slice.length ? slice : [{ t: "0", score: 0, label: "Start" }];
  }, [timeline, chartRevealIndex]);

  const visibleTraceSteps = traceSteps.slice(0, stepIndex + 1);

  useEffect(() => {
    if (!playing) return;
    const totalSteps = Math.max(traceSteps.length, timeline.length, 1);
    const interval = setInterval(() => {
      setStepIndex((i) => {
        const next = i + 1;
        if (next >= totalSteps) {
          setPlaying(false);
          return i;
        }
        return next;
      });
      setChartRevealIndex((i) => (timeline.length > 0 ? Math.min(i + 1, timeline.length - 1) : 0));
    }, 800);
    return () => clearInterval(interval);
  }, [playing, traceSteps.length, timeline.length]);

  const loadFromApi = useCallback(() => {
    setApiError(null);
    setLoadingFromApi(true);
    api
      .getFinancialDemo()
      .then((res) => {
        const riskSignals = res.output?.risk_signals ?? [];
        const stepTrace = res.output?.step_trace ?? [];
        if (stepTrace.length) {
          setScenarioSource("api_demo");
          setStepIndex(0);
          setChartRevealIndex(0);
        }
      })
      .catch((e) => setApiError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoadingFromApi(false));
  }, []);

  const reset = useCallback(() => {
    setPlaying(false);
    setStepIndex(0);
    setChartRevealIndex(0);
    setResetViewKey((k) => k + 1);
  }, []);

  if (graphLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-[520px] w-full rounded-2xl" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Scenario Replay</h1>
        <p className="text-muted-foreground text-sm mt-1">
          {title}. {description}
        </p>
        <div className="mt-2 rounded-lg border px-3 py-2 text-sm font-medium bg-muted/50">
          {alertId
            ? "Viewing alert — graph shows your household evidence; highlighted nodes are entities in this alert."
            : scenarioSource === "last_run"
              ? "Connected to your last run and open alerts. Graph is the same data as Graph view."
              : "Graph = same as Graph view. Use “Refresh from API” to run a demo pipeline."}
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-center">
        <Button className="rounded-xl" onClick={() => { reset(); setPlaying(true); }} disabled={playing}>
          <Play className="h-4 w-4 mr-2" />
          Play
        </Button>
        <Button variant="outline" className="rounded-xl" onClick={reset}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset
        </Button>
        {!alertId && (
          <Button
            variant="secondary"
            className="rounded-xl"
            onClick={loadFromApi}
            disabled={loadingFromApi}
          >
            <Download className="h-4 w-4 mr-2" />
            {loadingFromApi ? "Loading…" : "Refresh from API (run demo)"}
          </Button>
        )}
        {alertId && (
          <Link href={`/alerts/${alertId}`}>
            <Button variant="outline" size="sm" className="rounded-xl">
              <ExternalLink className="h-4 w-4 mr-2" />
              Open alert
            </Button>
          </Link>
        )}
        <Link href="/graph">
          <Button variant="ghost" size="sm" className="rounded-xl">
            Open Graph view
          </Button>
        </Link>
        {apiError && <span className="text-destructive text-sm">{apiError}</span>}
      </div>

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

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Risk score over time</CardTitle>
          <p className="text-muted-foreground text-sm">
            {alertId ? "This alert’s score" : "Recent open alerts (score by order)"}
          </p>
        </CardHeader>
        <CardContent>
          <div className="h-[220px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="t" tick={{ fontSize: 10 }} tickFormatter={(t) => `${t}`} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} width={28} />
                <Tooltip formatter={(v: number | undefined) => [v != null ? (v * 100).toFixed(0) + "%" : "—", "Score"]} />
                <Line type="monotone" dataKey="score" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={true} name="Score" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Evidence graph</CardTitle>
          <p className="text-muted-foreground text-sm">
            Same graph as Graph view. {highlightNodeIds.length > 0 ? "Highlighted nodes = entities in this alert. Toggle “Highlight alert nodes” to zoom to only those." : "Open with ?alert=id to highlight an alert’s entities."}
          </p>
        </CardHeader>
        <CardContent>
          {hasGraphData ? (
            <ScientificGraphCanvas
              nodes={nodes}
              edges={edges}
              highlightNodeIds={highlightNodeIds.length > 0 ? highlightNodeIds : undefined}
              resetViewKey={resetViewKey}
              layoutMode={layoutMode}
              nodeScaling={nodeScaling}
              edgeRenderMode={edgeRenderMode}
            />
          ) : (
            <div className="flex h-[520px] items-center justify-center rounded-2xl border border-dashed border-border bg-muted/30 text-muted-foreground text-sm">
              No graph data. Run a safety check or ingest events, or open Graph view.
            </div>
          )}
        </CardContent>
      </Card>

      {traceSteps.length > 0 && <AgentTrace steps={visibleTraceSteps} title="What happened" />}

      <AnimatePresence>
        {stepIndex < traceSteps.length && traceSteps[stepIndex] && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="rounded-2xl border border-primary/30 bg-primary/5 p-4"
          >
            <p className="text-sm font-medium">Current step: {traceSteps[stepIndex].step}</p>
            <p className="text-muted-foreground text-sm mt-1">{traceSteps[stepIndex].description}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
