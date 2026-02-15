"use client";

import React, { useRef, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  useAlertPage,
  useSubmitFeedback,
  useOutreachMutation,
  useDeepDiveExplainMutation,
  useIncidentResponseRunMutation,
  useCompletePlaybookTaskMutation,
  useAlertRefreshMutation,
} from "@/hooks/use-api";
import { IncidentHeader } from "@/components/alerts/IncidentHeader";
import { WhatChangedPanel } from "@/components/alerts/WhatChangedPanel";
import { EvidenceOverview } from "@/components/alerts/EvidenceOverview";
import { RingSnapshotCard } from "@/components/alerts/RingSnapshotCard";
import { DecisionPanel } from "@/components/alerts/DecisionPanel";
import { AnalysisDetailsCollapsible } from "@/components/alerts/AnalysisDetailsCollapsible";
import { PolicyGateCard } from "@/components/policy-gate-card";
import { Skeleton } from "@/components/ui/skeleton";
import { useAppStore } from "@/store/use-app-store";
import type { TraceStep } from "@/components/agent-trace";
import type { SubgraphNode, SubgraphEdge } from "@/lib/api/schemas";

const defaultTraceSteps: TraceStep[] = [
  { step: "Ingest", description: "Loaded events from session.", inputs: "session_ids", outputs: "events", status: "success", latency_ms: 12 },
  { step: "Normalize", description: "Extracted utterances, entities, mentions.", outputs: "utterances, entities", status: "success", latency_ms: 45 },
  { step: "GraphUpdate", description: "Updated household graph.", status: "success", latency_ms: 28 },
  { step: "Score", description: "Ran risk model.", outputs: "risk_score", status: "success", latency_ms: 120 },
  { step: "Explain", description: "Generated motifs and evidence subgraph.", outputs: "motifs, subgraph", status: "success", latency_ms: 85 },
  { step: "ConsentGate", description: "Checked consent for escalation and watchlist.", status: "success", latency_ms: 2 },
  { step: "Watchlist", description: "Synthesized watchlist entries.", status: "success", latency_ms: 15 },
  { step: "EscalationDraft", description: "Drafted caregiver notification (not sent).", status: "success", latency_ms: 8 },
  { step: "Persist", description: "Saved risk signal and watchlist.", status: "success", latency_ms: 35 },
];

export function AlertDetailContent({ id }: { id: string }) {
  const demoMode = useAppStore((s) => s.demoMode);
  const judgeMode = useAppStore((s) => s.judgeMode);
  const setJudgeMode = useAppStore((s) => s.setJudgeMode);
  const decisionPanelRef = useRef<HTMLDivElement>(null);

  const { data: page, isLoading } = useAlertPage(id);
  const detail = page?.risk_signal_detail;
  const similarData = page?.similar_incidents ?? null;
  const events = page?.session_events ?? [];
  const outreachForThisSignal = page?.outreach_actions ?? [];
  const playbook = page?.playbook ?? null;
  const playbookLoading = isLoading;
  const capabilities = page?.capabilities_snapshot ?? null;
  const canTriggerOutreach = page?.investigation_refresh_allowed ?? false;
  const canRefreshInvestigation = page?.investigation_refresh_allowed ?? false;
  const hasLockCardCap = (capabilities?.bank_control_capabilities as Record<string, unknown>)?.lock_card === true;

  const submitFeedback = useSubmitFeedback(id);
  const outreachMutation = useOutreachMutation();
  const deepDiveMutation = useDeepDiveExplainMutation(id);
  const [graphView, setGraphView] = React.useState<"model" | "deep_dive">("model");
  const [notes, setNotes] = React.useState("");
  const [outreachPreview, setOutreachPreview] = React.useState<{ caregiver_message?: string; elder_safe_message?: string } | null>(null);
  const [outreachConfirm, setOutreachConfirm] = React.useState(false);
  const incidentResponseMutation = useIncidentResponseRunMutation(id);
  const completeTaskMutation = useCompletePlaybookTaskMutation(playbook?.id ?? "", id);
  const [refreshMessage, setRefreshMessage] = React.useState<string | null>(null);
  const alertRefreshMut = useAlertRefreshMutation(id);

  const scrollToDecisionPanel = useCallback(() => {
    decisionPanelRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  if (isLoading || !page || !detail) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  const expl = detail.explanation as Record<string, unknown>;
  const modelAvailable = expl.model_available as boolean | undefined;
  const redacted = expl.redacted as boolean | undefined;
  const redactionReason = expl.redaction_reason as string | undefined;
  const evidenceQuality = expl.model_evidence_quality as { sparsity?: number; edges_kept?: number; edges_total?: number } | undefined;
  const motifs = (expl.motif_tags as string[]) ?? (expl.motifs as string[]) ?? [];
  const recommended = (detail.recommended_action as Record<string, unknown>) ?? {};
  const steps = (recommended.checklist as string[]) ?? (recommended.steps as string[]) ?? [];
  const traceSteps = (expl.logs as TraceStep[] | undefined)
    ? (expl.logs as TraceStep[]).map((l) => ({
        ...l,
        status: (l.status as TraceStep["status"]) ?? "success",
      }))
    : defaultTraceSteps;

  const summaryLine = (expl.narrative as string) || (expl.summary as string) || null;
  const whatChangedSummary = expl.what_changed_summary as string | undefined;
  const decisionRuleUsed = expl.decision_rule_used as string | undefined;
  const calibratedP = expl.calibrated_p as number | undefined;
  const ruleScore = expl.rule_score as number | undefined;
  const fusionScore = expl.fusion_score as number | undefined;
  const structuralMotifs = expl.structural_motifs ?? expl.motifs;
  const rawDeep = expl.deep_dive_subgraph as { nodes?: { id: string; type?: string; label?: string | null; score?: number | null }[]; edges?: { src: string; dst: string; type?: string; weight?: number; importance?: number; rank?: number }[] } | undefined;
  const deepDiveSubgraph: { nodes: SubgraphNode[]; edges: SubgraphEdge[] } | null =
    rawDeep?.nodes?.length
      ? {
          nodes: rawDeep.nodes.map((n) => ({ id: String(n.id), type: n.type ?? "entity", label: n.label ?? null, score: n.score ?? null })),
          edges: (rawDeep.edges ?? []).map((e) => ({ src: e.src, dst: e.dst, type: e.type ?? "", weight: e.weight ?? e.importance ?? null, rank: e.rank ?? null })),
        }
      : null;
  const subgraphNodes = detail.subgraph?.nodes ?? [];
  const subgraphEdges = detail.subgraph?.edges ?? [];

  const handlePreviewMessage = () => {
    outreachMutation.mutate(
      { risk_signal_id: id, dry_run: true },
      {
        onSuccess: (res) => {
          const preview = res.preview as { caregiver_message?: string; elder_safe_message?: string } | undefined;
          const suppressed = (res as { suppressed?: boolean }).suppressed;
          const looksLikeInternalNotes = (s: string) => /signal loaded|consent_allow|contacts=\d|step_trace/i.test(s);
          const isBareChannel = (s: string) => /^(sms|email|voice_call)$/i.test(s.trim());
          const okToShow = preview?.caregiver_message?.trim() && !looksLikeInternalNotes(preview.caregiver_message.trim()) && !isBareChannel(preview.caregiver_message);
          if (okToShow) {
            setOutreachPreview({
              caregiver_message: preview.caregiver_message,
              elder_safe_message: preview.elder_safe_message ?? "We've shared a brief summary with your caregiver.",
            });
          } else {
            setOutreachPreview({
              caregiver_message: suppressed
                ? "Outbound contact is currently off in settings, so no message would be sent. Enable it to see a draft of the alert here."
                : "The caregiver would receive a short summary of this alert and a link to the dashboard.",
              elder_safe_message: "We've shared a brief summary with your caregiver.",
            });
          }
        },
        onError: (err) => console.error("Outreach preview failed:", err),
      }
    );
  };

  const handleConfirmSend = () => {
    setOutreachConfirm(true);
    outreachMutation.mutate(
      { risk_signal_id: id, dry_run: false },
      {
        onSuccess: (res) => {
          const suppressed = (res as { suppressed?: boolean }).suppressed;
          if (suppressed) {
            setOutreachPreview({
              caregiver_message: "This alert wasn't sent: outbound contact is turned off in household settings.",
              elder_safe_message: "No message was sent (household settings).",
            });
          } else {
            setOutreachPreview(null);
          }
          setOutreachConfirm(false);
        },
        onError: (err) => {
          console.error("Outreach send failed:", err);
          setOutreachConfirm(false);
        },
      }
    );
  };

  return (
    <div className="space-y-6">
      {refreshMessage && (
        <div className="rounded-xl border border-green-500/40 bg-green-500/10 px-4 py-2 text-sm text-green-800 dark:text-green-200">
          {refreshMessage}
        </div>
      )}
      {redacted && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-800 dark:text-amber-200">
          {redactionReason ?? "Some content is redacted due to consent settings. Raw utterance text and entity labels are withheld."}
        </div>
      )}

      <IncidentHeader
        signalId={id}
        signalType={detail.signal_type}
        status={detail.status}
        severity={detail.severity}
        score={detail.score}
        ts={detail.ts}
        summaryLine={summaryLine}
        decisionRuleUsed={decisionRuleUsed}
        canRefreshInvestigation={canRefreshInvestigation}
        canNotifyCaregiver={canTriggerOutreach}
        demoMode={demoMode}
        onRefreshInvestigation={() => {
          alertRefreshMut.mutate(undefined, {
            onSuccess: () => {
              setRefreshMessage("Investigation refreshed");
              setTimeout(() => setRefreshMessage(null), 3000);
            },
          });
        }}
        onNotifyCaregiver={scrollToDecisionPanel}
        refreshPending={alertRefreshMut.isPending}
        ringId={(expl.ring_id as string) || null}
        judgeMode={judgeMode}
        onJudgeModeChange={setJudgeMode}
      />

      <WhatChangedPanel
        whatChangedSummary={whatChangedSummary}
        ringOrGraphSummary={
          detail.subgraph && (subgraphNodes.length > 0 || subgraphEdges.length > 0)
            ? `${subgraphNodes.length} entities in evidence subgraph. ${modelAvailable === false ? "Rule/motif only (model not run)." : ""}`
            : null
        }
        hasNoChanges={!whatChangedSummary && !detail.subgraph?.nodes?.length}
      />

      {detail.signal_type === "ring_candidate" && (expl.ring_id as string) && (
        <RingSnapshotCard ringId={expl.ring_id as string} />
      )}

      <EvidenceOverview
        events={events}
        subgraph={detail.subgraph ?? null}
        deepDiveSubgraph={deepDiveSubgraph}
        graphView={graphView}
        onGraphViewChange={setGraphView}
        modelAvailable={modelAvailable === true}
        onRequestDeepDive={() => deepDiveMutation.mutate("pg")}
        deepDivePending={deepDiveMutation.isPending}
        motifs={motifs}
        semanticTags={(expl.semantic_tags as string[]) ?? []}
        independenceViolation={expl.independence_violation as boolean | undefined}
        signalId={id}
      />

      <div id="decision-panel" ref={decisionPanelRef}>
        <DecisionPanel
          steps={steps}
          canNotifyCaregiver={canTriggerOutreach}
          demoMode={demoMode}
          onPreviewMessage={handlePreviewMessage}
          onConfirmSend={handleConfirmSend}
          previewPending={outreachMutation.isPending && !outreachPreview}
          sendPending={outreachMutation.isPending && !!outreachConfirm}
          outreachPreview={outreachPreview}
          onClearPreview={() => setOutreachPreview(null)}
          outreachActions={outreachForThisSignal}
          outreachError={outreachMutation.isError ? (outreachMutation.error as Error) : null}
          playbook={playbook ? { id: playbook.id, tasks: playbook.tasks ?? [] } : null}
          playbookLoading={playbookLoading}
          onRunIncidentResponse={() =>
            incidentResponseMutation.mutate(
              { risk_signal_id: id, dry_run: false },
              { onError: (err) => console.error("Incident response run failed:", err) }
            )
          }
          incidentResponsePending={incidentResponseMutation.isPending}
          incidentResponseError={incidentResponseMutation.isError ? (incidentResponseMutation.error as Error) : null}
          onCompleteTask={(taskId) => completeTaskMutation.mutate(taskId)}
          completeTaskPending={completeTaskMutation.isPending}
          hasLockCardCap={hasLockCardCap}
          capabilitiesNote={!hasLockCardCap && capabilities ? "Your bank integration does not support lock card; we prepared the call script instead." : undefined}
        />
      </div>

      <AnalysisDetailsCollapsible
        similarData={similarData}
        traceSteps={traceSteps}
        judgeMode={judgeMode}
        explanationJson={judgeMode ? (detail.explanation as Record<string, unknown>) : undefined}
        decisionRuleUsed={decisionRuleUsed}
        calibratedP={calibratedP}
        ruleScore={ruleScore}
        fusionScore={fusionScore}
        structuralMotifsJson={judgeMode ? structuralMotifs : undefined}
      />

      {demoMode && (
        <p className="text-amber-700 dark:text-amber-400 text-sm rounded-lg bg-amber-500/10 px-4 py-2 border border-amber-500/30">
          Demo mode is on. Turn it off (sidebar) and sign in to use Notify caregiver and Action plan.
        </p>
      )}

      <PolicyGateCard
        withheld={["Raw utterance text withheld until consent."]}
        couldShare={["Full transcript for caregiver review."]}
      />

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Feedback</CardTitle>
          <p className="text-muted-foreground text-sm">
            Confirm scam, false alarm, or unsure. Notes are stored for calibration.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Optional notes…"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            className="rounded-xl min-h-[80px]"
          />
          <div className="flex flex-wrap gap-2">
            <Button
              variant="destructive"
              className="rounded-xl"
              disabled={submitFeedback.isPending}
              onClick={() =>
                submitFeedback.mutate(
                  { label: "true_positive", notes: notes || undefined },
                  { onSuccess: () => setNotes("") }
                )
              }
            >
              Confirm scam
            </Button>
            <Button
              variant="outline"
              className="rounded-xl"
              disabled={submitFeedback.isPending}
              onClick={() =>
                submitFeedback.mutate(
                  { label: "false_positive", notes: notes || undefined },
                  { onSuccess: () => setNotes("") }
                )
              }
            >
              False alarm
            </Button>
            <Button
              variant="secondary"
              className="rounded-xl"
              disabled={submitFeedback.isPending}
              onClick={() =>
                submitFeedback.mutate(
                  { label: "unsure", notes: notes || undefined },
                  { onSuccess: () => setNotes("") }
                )
              }
            >
              Unsure
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
