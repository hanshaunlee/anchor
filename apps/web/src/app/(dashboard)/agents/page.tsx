"use client";

import { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TraceViewer } from "@/components/trace-viewer";
import { AgentTrace } from "@/components/agent-trace";
import { ArtifactLinks } from "@/components/artifact-links";
import { RunResultSummary } from "@/components/run-result-summary";
import { ExplainableIds } from "@/components/explainable-ids";
import { ConsentGateBanner } from "@/components/consent-gate-banner";
import { ModelHealthCard } from "@/components/model-health-card";
import { ActionsHistory, type ActionRow } from "@/components/actions-history";
import {
  useHouseholdMe,
  useAgentsStatus,
  useAgentCatalog,
  useInvestigationRunMutation,
  useMaintenanceRunMutation,
  useAgentTrace,
  useOutreachSummary,
  useOutreachHistory,
  useOutreachCandidates,
  useOutreachPreviewMutation,
  useOutreachSendMutation,
  useAgentRunMutation,
} from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import { AgentStrip } from "@/components/agent-strip";
import { ImplementedAgentsCard } from "@/components/implemented-agents-card";
import {
  Bot,
  Play,
  Search,
  Activity,
  Wrench,
  Send,
  Eye,
  ShieldCheck,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useState as useReactState } from "react";

const PIPELINE_STEPS = [
  "Ingest",
  "GraphUpdate",
  "Score",
  "Explain",
  "ConsentGate",
  "Watchlist",
  "EscalationDraft",
  "Persist",
];

const EXPECTED_PROCESS_STEPS: Array<{ step: string; description: string }> = [
  { step: "Load context", description: "Loading household and calibration…" },
  { step: "Normalize events", description: "Processing calls and messages…" },
  { step: "Run financial detection", description: "Detecting risks and motifs…" },
  { step: "Ensure narratives", description: "Adding plain-language explanations…" },
  { step: "Optional ring discovery", description: "Checking contact clusters…" },
  { step: "Finalize and broadcast", description: "Saving and notifying…" },
  { step: "Recurring contacts", description: "Updating watchlist…" },
];

export default function AutomationCenterPage() {
  const [timeWindowDays, setTimeWindowDays] = useState(7);
  const [useDemoEvents, setUseDemoEvents] = useState(false);
  const [lastInvestigationResult, setLastInvestigationResult] = useState<{
    supervisor_run_id: string | null;
    created_signal_ids: string[];
    updated_signal_ids?: string[];
    created_watchlist_ids?: string[];
    outreach_candidates?: Array<{ risk_signal_id?: string; severity?: number; decision_rule_used?: string }>;
    summary_json: { counts?: Record<string, number>; warnings?: string[]; message?: string };
    step_trace: Array<{ step?: string; status?: string; notes?: string }>;
  } | null>(null);
  const [selectedSignalForOutreach, setSelectedSignalForOutreach] = useState<string | null>(null);
  const [devPasteInput, setDevPasteInput] = useState("");
  const [activeTab, setActiveTab] = useState("investigation");
  const [maintenanceConfirmOpen, setMaintenanceConfirmOpen] = useState(false);
  const [sendConfirmSignal, setSendConfirmSignal] = useState<string | null>(null);
  const [historyStatusFilter, setHistoryStatusFilter] = useState("all");
  const [showManualOutreach, setShowManualOutreach] = useState(false);

  const demoMode = useAppStore((s) => s.demoMode);
  const explainMode = useAppStore((s) => s.explainMode);
  const [advancedOptionsOpen, setAdvancedOptionsOpen] = useReactState(false);
  const [technicalTraceOpen, setTechnicalTraceOpen] = useReactState(explainMode);
  const { data: me } = useHouseholdMe();
  const role = me?.role ?? "elder";
  const isAdmin = role === "admin";
  const isCaregiverOrAdmin = role === "caregiver" || role === "admin";
  const canSeeAutomation = isCaregiverOrAdmin;
  const canSeeDevTools = isAdmin;

  const { data: catalogData } = useAgentCatalog();
  const { data: statusData, isLoading: statusLoading } = useAgentsStatus();
  const investigationMut = useInvestigationRunMutation();
  const maintenanceMut = useMaintenanceRunMutation();
  const outreachPreviewMut = useOutreachPreviewMutation();
  const outreachSendMut = useOutreachSendMutation();
  const { data: outreachSummary } = useOutreachSummary(canSeeAutomation);
  const { data: outreachHistory } = useOutreachHistory({ limit: 50 });
  const { data: outreachCandidatesData } = useOutreachCandidates(canSeeAutomation);

  const supervisorStatus = statusData?.agents?.find((a: { agent_name?: string }) => a.agent_name === "supervisor");
  const modelHealthStatus = statusData?.agents?.find((a: { agent_name?: string }) => a.agent_name === "model_health");
  const supervisorRunId = lastInvestigationResult?.supervisor_run_id ?? supervisorStatus?.last_run_id;
  const { data: supervisorTrace } = useAgentTrace(supervisorRunId ?? null, "supervisor");

  const investigationStepTrace =
    (lastInvestigationResult?.step_trace?.length
      ? lastInvestigationResult.step_trace
      : (supervisorTrace?.step_trace as Array<{ step?: string; status?: string; error?: string; notes?: string }> | undefined)) ?? [];

  const runInvestigation = (dryRun: boolean) => {
    if (demoMode) {
      setLastInvestigationResult({
        supervisor_run_id: "demo-supervisor-run-1",
        created_signal_ids: ["0b42590f-2584-4d5a-9d16-2360a9c019b8", "88b02fe7-8628-436d-aeff-7208fca628f3"],
        summary_json: { counts: { new_signals: 2, watchlists: 1, outreach_candidates: 2 } },
        step_trace: [
          { step: "Ingest", status: "success", notes: "Demo: mock events loaded" },
          { step: "GraphUpdate", status: "success", notes: "Demo: graph updated" },
          { step: "Score", status: "success", notes: "Demo: risk signals scored" },
          { step: "Explain", status: "success", notes: "Demo: narratives attached" },
          { step: "ConsentGate", status: "success", notes: "Demo: consent checked" },
          { step: "Watchlist", status: "success", notes: "Demo: watchlists updated" },
          { step: "EscalationDraft", status: "success", notes: "Demo: outreach candidates created" },
          { step: "Persist", status: "success", notes: "Demo: artifacts saved" },
        ],
      });
      return;
    }
    investigationMut.mutate(
      { time_window_days: timeWindowDays, dry_run: dryRun, use_demo_events: useDemoEvents },
      {
        onSuccess: (res) => {
          const enqueued = (res as { enqueued?: boolean }).enqueued;
          if (enqueued) {
            setLastInvestigationResult({
              supervisor_run_id: null,
              created_signal_ids: [],
              updated_signal_ids: [],
              created_watchlist_ids: [],
              outreach_candidates: [],
              summary_json: { message: (res as { message?: string }).message ?? "Queued. Modal or worker will run it. Refresh in a minute." },
              step_trace: [],
            });
            return;
          }
          setLastInvestigationResult({
            supervisor_run_id: res.supervisor_run_id ?? null,
            created_signal_ids: res.created_signal_ids ?? [],
            updated_signal_ids: res.updated_signal_ids,
            created_watchlist_ids: res.created_watchlist_ids,
            outreach_candidates: res.outreach_candidates,
            summary_json: res.summary_json ?? {},
            step_trace: res.step_trace ?? [],
          });
        },
      }
    );
  };

  const runInvestigationInBackground = () => {
    if (demoMode) return;
    investigationMut.mutate(
      { time_window_days: timeWindowDays, enqueue: true },
      {
        onSuccess: (res) => {
          const enqueued = (res as { enqueued?: boolean }).enqueued;
          setLastInvestigationResult({
            supervisor_run_id: null,
            created_signal_ids: [],
            updated_signal_ids: [],
            created_watchlist_ids: [],
            outreach_candidates: [],
            summary_json: {
              message: enqueued
                ? (res as { message?: string }).message ?? "Investigation queued. Modal will run it in ~2 min. Refresh the page."
                : (res as { message?: string }).message ?? "Already queued or skipped.",
            },
            step_trace: [],
          });
        },
      }
    );
  };

  if (!canSeeAutomation) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Bot className="h-7 w-7" />
          Agent Console
        </h1>
        <Card className="rounded-2xl shadow-sm border-muted">
          <CardContent className="pt-6">
            <p className="text-muted-foreground text-sm">
              Safety checks and notification tools are available to caregivers and administrators.
              You can view your alerts and elder-safe summary from the dashboard and Alerts.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Bot className="h-7 w-7" />
          Agent Console
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Run a safety check to detect risks, add plain-English explanations, and prepare next steps. Nothing is sent until you approve.
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="rounded-xl bg-muted p-1 gap-1">
          <TabsTrigger value="investigation" className="rounded-lg data-[state=active]:bg-background">
            <Search className="h-4 w-4 mr-2" />
            Investigation
          </TabsTrigger>
          <TabsTrigger value="actions" className="rounded-lg data-[state=active]:bg-background">
            <Send className="h-4 w-4 mr-2" />
            Notify / Next steps
          </TabsTrigger>
          <TabsTrigger value="system" className="rounded-lg data-[state=active]:bg-background">
            <Activity className="h-4 w-4 mr-2" />
            System Health
          </TabsTrigger>
          {canSeeDevTools && (
            <TabsTrigger value="dev" className="rounded-lg data-[state=active]:bg-background">
              <Wrench className="h-4 w-4 mr-2" />
              Advanced / Dev
            </TabsTrigger>
          )}
        </TabsList>

        {/* Tab 1: Investigation — Run Safety Check (family-friendly) + Advanced options + live Process on the right */}
        <TabsContent value="investigation" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
            <div className="space-y-6 min-w-0">
          <Card className="rounded-2xl shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Run a Safety Check</CardTitle>
              <p className="text-muted-foreground text-sm">
                One click: detect risks, add explanations, and prepare recommended next steps (not sent).
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  className="rounded-xl"
                  onClick={() => runInvestigation(false)}
                  disabled={investigationMut.isPending}
                >
                  <ShieldCheck className="h-4 w-4 mr-2" />
                  {investigationMut.isPending ? "Running…" : demoMode ? "Run Safety Check (demo)" : "Run Safety Check"}
                </Button>
                <button
                  type="button"
                  className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
                  onClick={() => setAdvancedOptionsOpen(!advancedOptionsOpen)}
                >
                  {advancedOptionsOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  Advanced
                </button>
              </div>
              {advancedOptionsOpen && (
                <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-muted/20 p-4">
                  <Button
                    variant="secondary"
                    size="sm"
                    className="rounded-xl"
                    onClick={() => runInvestigationInBackground()}
                    disabled={investigationMut.isPending}
                  >
                    <Play className="h-4 w-4 mr-2" />
                    {investigationMut.isPending ? "Queuing…" : "Run in background (Modal)"}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="rounded-xl"
                    onClick={() => runInvestigation(true)}
                    disabled={investigationMut.isPending}
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    Preview (dry run)
                  </Button>
                  {isAdmin && (
                    <label className="flex items-center gap-2 text-sm cursor-pointer">
                      <input
                        type="checkbox"
                        checked={useDemoEvents}
                        onChange={(e) => setUseDemoEvents(e.target.checked)}
                        className="rounded border-border"
                      />
                      Use demo events
                    </label>
                  )}
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <span>Time window:</span>
                    <select
                      className="rounded-lg border border-input bg-background px-2 py-1 text-sm"
                      value={timeWindowDays}
                      onChange={(e) => setTimeWindowDays(Number(e.target.value))}
                    >
                      {[1, 3, 7, 14, 30].map((d) => (
                        <option key={d} value={d}>{d} days</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Run result: empty state or canonical summary */}
          {statusLoading ? (
            <Skeleton className="h-48 w-full rounded-xl" />
          ) : !supervisorRunId && !lastInvestigationResult ? (
            <Card className="rounded-2xl shadow-sm border-dashed">
              <CardContent className="pt-6 pb-6">
                <p className="text-muted-foreground text-sm mb-4">No safety check run yet.</p>
                <Button className="rounded-xl" onClick={() => runInvestigation(false)} disabled={investigationMut.isPending}>
                  <ShieldCheck className="h-4 w-4 mr-2" />
                  Run Safety Check
                </Button>
              </CardContent>
            </Card>
          ) : (
            <RunResultSummary
              bundle={{
                created_signal_ids: lastInvestigationResult?.created_signal_ids ?? (supervisorStatus?.last_run_summary as Record<string, unknown> | undefined)?.created_signal_ids as string[] | undefined,
                updated_signal_ids: lastInvestigationResult?.updated_signal_ids ?? (supervisorStatus?.last_run_summary as Record<string, unknown> | undefined)?.updated_signal_ids as string[] | undefined,
                created_watchlist_ids: lastInvestigationResult?.created_watchlist_ids,
                outreach_candidates: lastInvestigationResult?.outreach_candidates ?? (supervisorStatus?.last_run_summary as Record<string, unknown> | undefined)?.outreach_candidates as Array<{ risk_signal_id?: string; severity?: number; decision_rule_used?: string }> | undefined,
                summary_json: lastInvestigationResult?.summary_json ?? (supervisorStatus?.last_run_summary as Record<string, unknown> | undefined)?.summary_json as { counts?: Record<string, number>; warnings?: string[] } | undefined,
                step_trace: investigationStepTrace.length ? investigationStepTrace : undefined,
              }}
              meta={{
                run_id: supervisorRunId ?? undefined,
                timestamp: supervisorStatus?.last_run_at ?? undefined,
                status: supervisorStatus?.last_run_status ?? undefined,
              }}
              onReviewOutreach={() => setActiveTab("actions")}
              onReviewWatchlists={() => {}}
              showTechnicalIds={explainMode}
            />
          )}

          {explainMode && (
            <>
              <ImplementedAgentsCard
                catalog={catalogData?.catalog ?? []}
                statusAgents={statusData?.agents}
              />
              {investigationStepTrace.length > 0 && (
                <AgentStrip
                  stepTrace={investigationStepTrace}
                  runId={supervisorRunId ?? undefined}
                  timestamp={supervisorStatus?.last_run_at ?? undefined}
                />
              )}
            </>
          )}

          {investigationStepTrace.length > 0 && (
            <details
              className="rounded-2xl border border-border bg-card"
              open={technicalTraceOpen}
              onToggle={(e) => setTechnicalTraceOpen((e.target as HTMLDetailsElement).open)}
            >
              <summary className="cursor-pointer px-4 py-3 font-medium text-sm flex items-center gap-2">
                {technicalTraceOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                Technical trace
              </summary>
              <div className="px-4 pb-4 pt-0">
                <TraceViewer stepTrace={investigationStepTrace} title="Pipeline steps" />
              </div>
            </details>
          )}

          {explainMode && supervisorRunId && (() => {
            const sum = supervisorStatus?.last_run_summary as Record<string, unknown> | undefined;
            const cnt = sum?.summary_json && typeof sum.summary_json === "object" ? (sum.summary_json as Record<string, unknown>).counts as Record<string, number> | undefined : undefined;
            return (
              <details className="rounded-2xl border border-border bg-card">
                <summary className="cursor-pointer px-4 py-3 font-medium text-sm">Artifacts</summary>
                <div className="px-4 pb-4 pt-0">
                  <ArtifactLinks
                    createdSignalIds={lastInvestigationResult?.created_signal_ids ?? (Array.isArray(sum?.created_signal_ids) ? sum.created_signal_ids as string[] : undefined)}
                    updatedSignalIds={lastInvestigationResult?.updated_signal_ids}
                    watchlistCount={lastInvestigationResult?.summary_json?.counts?.watchlists ?? cnt?.watchlists}
                    outreachCandidatesCount={lastInvestigationResult?.summary_json?.counts?.outreach_candidates ?? cnt?.outreach_candidates}
                  />
                </div>
              </details>
            );
          })()}
            </div>

            {/* Right: live process — what each agent is doing (gray when pending, then updates when done) */}
            <div className="lg:sticky lg:top-4 h-fit space-y-2">
              <Card className="rounded-2xl shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Process</CardTitle>
                  <p className="text-muted-foreground text-sm">
                    {investigationMut.isPending ? "Running… steps update below as they complete." : "What the safety check did."}
                  </p>
                </CardHeader>
                <CardContent>
                  {investigationMut.isPending ? (
                    <AgentTrace
                      steps={EXPECTED_PROCESS_STEPS.map((s) => ({ ...s, status: "pending" }))}
                      title=""
                    />
                  ) : investigationStepTrace.length > 0 ? (
                    <TraceViewer stepTrace={investigationStepTrace} title="Steps" />
                  ) : (
                    <p className="text-muted-foreground text-sm py-4">Run a safety check to see steps here.</p>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Tab 2: Actions */}
        <TabsContent value="actions" className="space-y-6 mt-4">
          <Card className="rounded-2xl shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Notify / Next steps</CardTitle>
              <p className="text-muted-foreground text-sm">
                Recommended message drafts: preview, then send with consent. Shown for high-severity candidates or from alert detail.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              {outreachSummary && (
                <div className="flex flex-wrap gap-4 text-sm">
                  <span className="rounded-lg bg-green-500/15 px-2 py-1 text-green-700 dark:text-green-400">Sent: {outreachSummary.counts?.sent ?? 0}</span>
                  <span className="rounded-lg bg-amber-500/15 px-2 py-1 text-amber-700 dark:text-amber-400">Suppressed: {outreachSummary.counts?.suppressed ?? 0}</span>
                  <span className="rounded-lg bg-destructive/15 px-2 py-1 text-destructive">Failed: {outreachSummary.counts?.failed ?? 0}</span>
                  <span className="rounded-lg bg-muted px-2 py-1">Queued: {outreachSummary.counts?.queued ?? 0}</span>
                </div>
              )}
              {outreachCandidatesData?.candidates && outreachCandidatesData.candidates.length > 0 && (
                <>
                  {!outreachCandidatesData.candidates[0]?.consent_ok && (
                    <ConsentGateBanner
                      message="Outreach is disabled until consent allows outbound contact and share with caregiver."
                      missingKeys={outreachCandidatesData.candidates[0]?.missing_consent_keys}
                    />
                  )}
                  {outreachCandidatesData.candidates[0] && !outreachCandidatesData.candidates[0].caregiver_contact_present && (
                    <ConsentGateBanner message="Add a caregiver contact in Settings to send notifications." missingKeys={["caregiver_contact"]} />
                  )}
                  <div className="space-y-3">
                    <p className="text-sm font-medium">Outreach candidates</p>
                    {outreachCandidatesData.candidates.map((c) => {
                      const blocked = (c.blocking_reasons ?? []).length > 0;
                      return (
                        <div key={c.risk_signal_id} className="rounded-xl border border-border bg-muted/20 p-4 flex flex-wrap items-center gap-3">
                          <div className="min-w-0 flex-1">
                            <Link href={`/alerts/${c.risk_signal_id}`} className="font-medium text-primary hover:underline">
                              Alert {c.risk_signal_id.slice(0, 8)}…
                            </Link>
                            <p className="text-xs text-muted-foreground mt-0.5">
                              Severity {c.severity ?? "—"} · {c.signal_type ?? "—"} · {c.candidate_reason ?? "trigger"}
                            </p>
                          </div>
                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              variant="outline"
                              disabled={blocked || outreachPreviewMut.isPending}
                              onClick={() =>
                                outreachPreviewMut.mutate(
                                  { risk_signal_id: c.risk_signal_id },
                                  {
                                    onSuccess: (data) => {
                                      alert("Preview:\n" + (data.preview?.caregiver_full ?? data.preview?.elder_safe ?? "No preview"));
                                    },
                                  }
                                )
                              }
                            >
                              Preview
                            </Button>
                            <Button
                              size="sm"
                              disabled={blocked || outreachSendMut.isPending}
                              onClick={() => setSendConfirmSignal(c.risk_signal_id)}
                            >
                              Send
                            </Button>
                          </div>
                          {blocked && (
                            <p className="text-xs text-amber-600 dark:text-amber-400 w-full">
                              Blocked: {(c.blocking_reasons ?? []).join(", ")}
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                  <ExplainableIds
                    context="alert_ids"
                    items={(outreachCandidatesData.candidates ?? []).map((c) => ({
                      id: c.risk_signal_id ?? "",
                      label: c.signal_type ? `Severity ${c.severity ?? "—"} · ${c.signal_type}` : undefined,
                    })).filter((it) => it.id)}
                    title="What these alerts are"
                    className="mt-3 pt-3 border-t border-border"
                  />
                </>
              )}
              {(!outreachCandidatesData?.candidates?.length) && (
                <p className="text-sm text-muted-foreground">No outreach candidates right now. Run Investigation to create candidates, or use the manual alert ID below.</p>
              )}
              <div className="pt-2 border-t border-border">
                <button
                  type="button"
                  className="text-sm text-muted-foreground hover:underline"
                  onClick={() => setShowManualOutreach(!showManualOutreach)}
                >
                  {showManualOutreach ? "Hide" : "Show"} manual alert ID (advanced)
                </button>
                {showManualOutreach && (
                  <div className="flex flex-wrap gap-2 items-center mt-2">
                    <input
                      type="text"
                      placeholder="Paste alert ID"
                      className="rounded-lg border border-input bg-background px-3 py-2 text-sm w-64"
                      value={selectedSignalForOutreach ?? ""}
                      onChange={(e) => setSelectedSignalForOutreach(e.target.value || null)}
                    />
                    <Button size="sm" variant="outline" disabled={!selectedSignalForOutreach || outreachPreviewMut.isPending} onClick={() => selectedSignalForOutreach && outreachPreviewMut.mutate({ risk_signal_id: selectedSignalForOutreach }, { onSuccess: (data) => alert("Preview:\n" + (data.preview?.caregiver_full ?? data.preview?.elder_safe ?? "No preview")) })}>Preview</Button>
                    <Button size="sm" disabled={!selectedSignalForOutreach || outreachSendMut.isPending} onClick={() => selectedSignalForOutreach && setSendConfirmSignal(selectedSignalForOutreach)}>Send</Button>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {sendConfirmSignal && (
            <Card className="rounded-2xl shadow-sm border-amber-500/40">
              <CardContent className="pt-6">
                <p className="text-sm mb-4">This will notify the caregiver for alert {sendConfirmSignal.slice(0, 8)}…. Confirm?</p>
                <div className="flex gap-2">
                  <Button size="sm" onClick={() => setSendConfirmSignal(null)} variant="outline">Cancel</Button>
                  <Button
                    size="sm"
                    onClick={() => {
                      outreachSendMut.mutate(
                        { risk_signal_id: sendConfirmSignal },
                        {
                          onSuccess: (data) => {
                            setSendConfirmSignal(null);
                            alert(data.sent ? "Sent." : data.suppressed ? "Suppressed (consent or policy)." : "Check status.");
                          },
                        }
                      );
                    }}
                    disabled={outreachSendMut.isPending}
                  >
                    {outreachSendMut.isPending ? "Sending…" : "Confirm send"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          <ActionsHistory
            actions={(outreachHistory?.actions ?? []) as ActionRow[]}
            statusFilter={historyStatusFilter}
            onStatusFilterChange={setHistoryStatusFilter}
          />
        </TabsContent>

        {/* Tab 3: System Health */}
        <TabsContent value="system" className="space-y-6 mt-4">
          <ModelHealthCard
            lastRun={modelHealthStatus ?? null}
            canRun={isAdmin}
            onRunMaintenance={() => setMaintenanceConfirmOpen(true)}
            isRunning={maintenanceMut.isPending}
          />
          {maintenanceConfirmOpen && (
            <Card className="rounded-2xl shadow-sm border-amber-500/40">
              <CardContent className="pt-6">
                <p className="text-sm mb-4">This may run drift and calibration. Red-team runs only in dev/staging. Continue?</p>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={() => setMaintenanceConfirmOpen(false)}>Cancel</Button>
                  <Button size="sm" onClick={() => { maintenanceMut.mutate(undefined, { onSettled: () => setMaintenanceConfirmOpen(false) }); }} disabled={maintenanceMut.isPending}>
                    {maintenanceMut.isPending ? "Running…" : "Run maintenance"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Tab 4: Advanced / Dev Tools */}
        {canSeeDevTools && (
          <TabsContent value="dev" className="space-y-6 mt-4">
            <div className="rounded-xl border border-amber-500/50 bg-amber-500/10 px-4 py-3 text-sm text-amber-800 dark:text-amber-200">
              <p className="font-medium">Admin tools</p>
              <p className="text-muted-foreground mt-1">These tools are for debugging and development. They can create test alerts and write to the database.</p>
            </div>

            <Card className="rounded-2xl shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Run individual agents</CardTitle>
                <p className="text-muted-foreground text-sm">
                  Investigation internals (narrative, ring). Model health internals (drift, calibration). Devtools (red-team).
                </p>
              </CardHeader>
              <CardContent>
                <DevAgentButtons catalog={catalogData} />
              </CardContent>
            </Card>

            <details className="rounded-2xl border border-border bg-card">
              <summary className="cursor-pointer px-4 py-3 font-medium text-sm">Debug: Pipeline steps</summary>
              <div className="px-4 pb-4 pt-1">
                <p className="text-muted-foreground text-sm mb-2">LangGraph: Ingest → GraphUpdate → Score → Explain → ConsentGate → Watchlist → EscalationDraft → Persist</p>
                <div className="flex flex-wrap gap-2">
                  {PIPELINE_STEPS.map((step) => (
                    <span key={step} className="rounded-xl bg-muted px-3 py-2 text-sm font-medium">{step}</span>
                  ))}
                </div>
              </div>
            </details>

            <details className="rounded-2xl border border-border bg-card">
              <summary className="cursor-pointer px-4 py-3 font-medium text-sm">Debug: Dry run (paste input)</summary>
              <div className="px-4 pb-4 pt-1">
                <p className="text-muted-foreground text-sm mb-2">Optional: paste JSON event batch. Preview runs on server.</p>
                <Textarea
                  placeholder='[{"event_type": "final_asr", "payload": {"text": "..."}}]'
                  value={devPasteInput}
                  onChange={(e) => setDevPasteInput(e.target.value)}
                  className="min-h-[100px] rounded-xl font-mono text-sm"
                />
              </div>
            </details>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}

function DevAgentButtons({ catalog }: { catalog?: { catalog?: Array<{ slug?: string; runnable?: boolean; reason?: string | null }> } }) {
  const slugs = ["drift", "narrative", "ring", "calibration", "redteam"] as const;
  const driftMut = useAgentRunMutation("drift");
  const narrativeMut = useAgentRunMutation("narrative");
  const ringMut = useAgentRunMutation("ring");
  const calibrationMut = useAgentRunMutation("calibration");
  const redteamMut = useAgentRunMutation("redteam");
  const mutMap = { drift: driftMut, narrative: narrativeMut, ring: ringMut, calibration: calibrationMut, redteam: redteamMut };
  const labels: Record<string, string> = { drift: "Drift", narrative: "Narrative", ring: "Ring", calibration: "Calibration", redteam: "Red-Team" };
  const slugToReason = (slug: string) => catalog?.catalog?.find((e) => e.slug === slug)?.reason ?? null;

  return (
    <div className="flex flex-wrap gap-4">
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-muted-foreground w-full">Investigation internals</span>
        {(["narrative", "ring"] as const).map((slug) => {
          const mut = mutMap[slug];
          const reason = slugToReason(slug);
          const disabled = mut.isPending || !!reason;
          return (
            <div key={slug} className="flex gap-2 items-center" title={reason ?? undefined}>
              <Button size="sm" variant="outline" disabled={disabled} onClick={() => mut.mutate({ dry_run: true })}>{mut.isPending ? "…" : `${labels[slug]} (dry)`}</Button>
              <Button size="sm" disabled={disabled} onClick={() => mut.mutate({ dry_run: false })}>Run {labels[slug]}</Button>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-muted-foreground w-full">Model health internals</span>
        {(["drift", "calibration"] as const).map((slug) => {
          const mut = mutMap[slug];
          const reason = slugToReason(slug);
          const disabled = mut.isPending || !!reason;
          return (
            <div key={slug} className="flex gap-2 items-center" title={reason ?? undefined}>
              <Button size="sm" variant="outline" disabled={disabled} onClick={() => mut.mutate({ dry_run: true })}>{mut.isPending ? "…" : `${labels[slug]} (dry)`}</Button>
              <Button size="sm" disabled={disabled} onClick={() => mut.mutate({ dry_run: false })}>Run {labels[slug]}</Button>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-muted-foreground w-full">Devtools</span>
        {(["redteam"] as const).map((slug) => {
          const mut = mutMap[slug];
          const reason = slugToReason(slug);
          const disabled = mut.isPending || !!reason;
          return (
            <div key={slug} className="flex gap-2 items-center" title={reason ?? undefined}>
              <Button size="sm" variant="outline" disabled={disabled} onClick={() => mut.mutate({ dry_run: true })}>{mut.isPending ? "…" : `${labels[slug]} (dry)`}</Button>
              <Button size="sm" disabled={disabled} onClick={() => mut.mutate({ dry_run: false })}>Run {labels[slug]}</Button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
