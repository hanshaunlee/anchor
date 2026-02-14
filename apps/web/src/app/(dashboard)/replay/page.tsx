"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { GraphEvidence } from "@/components/graph-evidence";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
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
import { Play, RotateCcw, Download } from "lucide-react";
import { api } from "@/lib/api";

type ReplayData = {
  title: string;
  description: string;
  risk_score_timeline: { t: number; score: number; label: string }[];
  subgraph_highlight_order: string[];
  agent_trace: TraceStep[];
};

const DEFAULT_REPLAY: ReplayData = {
  title: "Medicare scam scenario",
  description: "A synthetic scam session: unknown caller, urgency topic, then request for SSN.",
  risk_score_timeline: [
    { t: 0, score: 0.1, label: "Session start" },
    { t: 5, score: 0.25, label: "Mention of Medicare" },
    { t: 10, score: 0.45, label: "Call log intent" },
    { t: 60, score: 0.72, label: "Sensitive request (SSN)" },
    { t: 65, score: 0.82, label: "Signal raised" },
  ],
  subgraph_highlight_order: ["p1", "u1", "e1", "u2", "e2"],
  agent_trace: [
    { step: "Ingest", description: "Loaded 5 events from session.", inputs: "session_id, 5 events", outputs: "ingested_events", status: "success", latency_ms: 12 },
    { step: "Normalize", description: "Extracted utterances, entities, and mentions.", inputs: "5 events", outputs: "3 utterances, 2 entities", status: "success", latency_ms: 45 },
    { step: "GraphUpdate", description: "Updated household graph with new nodes and edges.", inputs: "entities, mentions", outputs: "graph_updated", status: "success", latency_ms: 28 },
    { step: "Score", description: "Ran risk model; score crossed threshold.", inputs: "graph", outputs: "risk_score 0.82", status: "success", latency_ms: 120 },
    { step: "Explain", description: "Generated motifs and evidence subgraph.", inputs: "risk_scores", outputs: "motifs: New contact, Urgency topic, Sensitive request", status: "success", latency_ms: 85 },
    { step: "ConsentGate", description: "Consent allows escalation and watchlist.", inputs: "consent_state", outputs: "allowed", status: "success", latency_ms: 2 },
    { step: "Watchlist", description: "Synthesized 1 watchlist entry for high-risk entity.", inputs: "risk_scores", outputs: "1 watchlist", status: "success", latency_ms: 15 },
    { step: "EscalationDraft", description: "Drafted caregiver notification (not sent).", inputs: "high_risk_count: 1", outputs: "draft", status: "success", latency_ms: 8 },
    { step: "Persist", description: "Saved risk signal and watchlist to database.", inputs: "signal, watchlist", outputs: "persisted", status: "success", latency_ms: 35 },
  ],
};

const SUBGRAPH_FROM_FIXTURE = {
  nodes: [
    { id: "p1", type: "person", label: "Unknown caller", score: 0.9 },
    { id: "e1", type: "entity", label: "Medicare", score: 0.85 },
    { id: "e2", type: "intent", label: "sensitive_request", score: 0.95 },
    { id: "u1", type: "utterance", label: "Someone called about Medicare", score: 0.7 },
    { id: "u2", type: "utterance", label: "They said I need to give my SSN", score: 0.88 },
  ],
  edges: [
    { src: "p1", dst: "u1", type: "mentioned_in", weight: 0.8, rank: 1 },
    { src: "e1", dst: "u1", type: "topic", weight: 0.9, rank: 1 },
    { src: "u1", dst: "u2", type: "follows", weight: 1, rank: 1 },
    { src: "u2", dst: "e2", type: "expresses", weight: 0.95, rank: 1 },
  ],
};

function logsToTraceSteps(logs: string[]): TraceStep[] {
  return logs.map((line, i) => ({
    step: `Step ${i + 1}`,
    description: line,
    status: "success" as const,
  }));
}

export default function ReplayPage() {
  const [replayData, setReplayData] = useState<ReplayData | null>(null);
  const [playing, setPlaying] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [chartRevealIndex, setChartRevealIndex] = useState(0);
  const [loadingFromApi, setLoadingFromApi] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Prefer real backend: try API first; fallback to static demo only if API fails (e.g. offline).
  useEffect(() => {
    api
      .getFinancialDemo()
      .then((res) => {
        const logs = res.output?.logs ?? [];
        const riskSignals = res.output?.risk_signals ?? [];
        const traceSteps = logs.length > 0 ? logsToTraceSteps(logs) : DEFAULT_REPLAY.agent_trace;
        const riskScoreTimeline =
          riskSignals.length > 0
            ? riskSignals.map((s: unknown, i: number) => {
                const score = typeof (s as { score?: number }).score === "number" ? (s as { score: number }).score : 0.5;
                return { t: i * 20, score, label: `Signal ${i + 1}` };
              })
            : DEFAULT_REPLAY.risk_score_timeline;
        if (riskScoreTimeline.length === 0) {
          riskScoreTimeline.push({ t: 0, score: 0.5, label: "Pipeline run" });
        }
        setReplayData({
          title: "Demo from API",
          description: res.input_summary ?? "Financial Security Agent run on demo events (no auth).",
          risk_score_timeline: riskScoreTimeline,
          subgraph_highlight_order: DEFAULT_REPLAY.subgraph_highlight_order,
          agent_trace: traceSteps,
        });
      })
      .catch(() => setReplayData(DEFAULT_REPLAY));
  }, []);

  const loadFromApi = () => {
    setApiError(null);
    setLoadingFromApi(true);
    api
      .getFinancialDemo()
      .then((res) => {
        const logs = res.output?.logs ?? [];
        const riskSignals = res.output?.risk_signals ?? [];
        const traceSteps = logs.length > 0 ? logsToTraceSteps(logs) : DEFAULT_REPLAY.agent_trace;
        const riskScoreTimeline =
          riskSignals.length > 0
            ? riskSignals.map((s: unknown, i: number) => {
                const score = typeof (s as { score?: number }).score === "number" ? (s as { score: number }).score : 0.5;
                return { t: i * 20, score, label: `Signal ${i + 1}` };
              })
            : DEFAULT_REPLAY.risk_score_timeline;
        if (riskScoreTimeline.length === 0) {
          riskScoreTimeline.push({ t: 0, score: 0.5, label: "Pipeline run" });
        }
        setReplayData({
          title: "Demo from API",
          description: res.input_summary ?? "Financial Security Agent run on demo events (no auth).",
          risk_score_timeline: riskScoreTimeline,
          subgraph_highlight_order: DEFAULT_REPLAY.subgraph_highlight_order,
          agent_trace: traceSteps,
        });
        setStepIndex(0);
        setChartRevealIndex(0);
      })
      .catch((e) => setApiError(e instanceof Error ? e.message : "Failed to load from API"))
      .finally(() => setLoadingFromApi(false));
  };

  const data = replayData ?? DEFAULT_REPLAY;
  const timeline = useMemo(() => data.risk_score_timeline ?? [], [data.risk_score_timeline]);
  const chartData = useMemo(
    () => timeline.slice(0, chartRevealIndex + 1).map((p) => ({ ...p, t: String(p.t) })),
    [timeline, chartRevealIndex]
  );
  const traceSteps = data.agent_trace ?? [];
  const visibleTraceSteps = traceSteps.slice(0, stepIndex + 1);
  const highlightIds = data.subgraph_highlight_order?.slice(0, stepIndex + 1) ?? [];

  useEffect(() => {
    if (!playing) return;
    const totalSteps = Math.max(traceSteps.length, timeline.length);
    const interval = setInterval(() => {
      setStepIndex((i) => {
        const next = i + 1;
        if (next >= totalSteps) {
          setPlaying(false);
          return i;
        }
        return next;
      });
      setChartRevealIndex((i) => Math.min(i + 1, timeline.length - 1));
    }, 800);
    return () => clearInterval(interval);
  }, [playing, traceSteps.length, timeline.length]);

  const reset = () => {
    setPlaying(false);
    setStepIndex(0);
    setChartRevealIndex(0);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Scenario Replay</h1>
        <p className="text-muted-foreground text-sm mt-1">
          {data.title}. {data.description}
        </p>
      </div>

      <div className="flex flex-wrap gap-3 items-center">
        <Button
          className="rounded-xl"
          onClick={() => {
            reset();
            setPlaying(true);
          }}
          disabled={playing}
        >
          <Play className="h-4 w-4 mr-2" />
          Play
        </Button>
        <Button variant="outline" className="rounded-xl" onClick={reset}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset
        </Button>
        <Button
          variant="secondary"
          className="rounded-xl"
          onClick={loadFromApi}
          disabled={loadingFromApi}
        >
          <Download className="h-4 w-4 mr-2" />
          {loadingFromApi ? "Loading…" : "Refresh from API (real demo run)"}
        </Button>
        {apiError && (
          <span className="text-destructive text-sm">{apiError}</span>
        )}
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Risk score over time</CardTitle>
          <p className="text-muted-foreground text-sm">Score rises as the story unfolds</p>
        </CardHeader>
        <CardContent>
          <div className="h-[220px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="t"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(t) => `${t}s`}
                />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} width={28} />
                <Tooltip formatter={(v: number | undefined) => [v != null ? (v * 100).toFixed(0) + "%" : "—", "Score"]} />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="hsl(var(--chart-1))"
                  strokeWidth={2}
                  dot={true}
                  name="Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <GraphEvidence
        variant="replay"
        nodes={SUBGRAPH_FROM_FIXTURE.nodes}
        edges={SUBGRAPH_FROM_FIXTURE.edges}
        highlightIds={highlightIds}
      />

      <AgentTrace steps={visibleTraceSteps} />

      <AnimatePresence>
        {stepIndex < traceSteps.length && traceSteps[stepIndex] && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="rounded-2xl border border-primary/30 bg-primary/5 p-4"
          >
            <p className="text-sm font-medium">Current step: {traceSteps[stepIndex].step}</p>
            <p className="text-muted-foreground text-sm mt-1">
              {traceSteps[stepIndex].description}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
