"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
import { PolicyGateCard } from "@/components/policy-gate-card";
import { Bot } from "lucide-react";
import { motion } from "framer-motion";

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

const MOCK_TRACE: TraceStep[] = [
  { step: "Ingest", description: "Load events for household in time range.", inputs: "session_id, 5 events", outputs: "ingested_events", status: "success", latency_ms: 12 },
  { step: "Normalize", description: "Build utterances, entities, mentions from events.", outputs: "3 utterances, 2 entities", status: "success", latency_ms: 45 },
  { step: "GraphUpdate", description: "Persist entities/mentions/relationships; mark graph_updated.", status: "success", latency_ms: 28 },
  { step: "Score", description: "Run GNN risk scoring; append risk_scores.", outputs: "risk_score 0.82", status: "success", latency_ms: 120 },
  { step: "Explain", description: "Generate motifs and evidence subgraph.", outputs: "motifs, subgraph", status: "success", latency_ms: 85 },
  { step: "ConsentGate", description: "Check consent_state; set consent_allows_escalation.", outputs: "allowed", status: "success", latency_ms: 2 },
  { step: "Watchlist", description: "Synthesize watchlist patterns if consent allows.", outputs: "1 watchlist", status: "success", latency_ms: 15 },
  { step: "EscalationDraft", description: "Draft caregiver notification (never sends).", outputs: "draft", status: "success", latency_ms: 8 },
  { step: "Persist", description: "Write risk_signals, watchlists to DB.", status: "success", latency_ms: 35 },
];

export default function AgentsPage() {
  const [dryRunInput, setDryRunInput] = useState("");
  const [lastRunLogs, setLastRunLogs] = useState<TraceStep[]>(MOCK_TRACE);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Bot className="h-7 w-7" />
          Agent Center
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          LangGraph pipeline: human-friendly view. Dry run with transcript or JSON event batch.
        </p>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Pipeline steps</CardTitle>
          <p className="text-muted-foreground text-sm">
            Ingest → GraphUpdate → Score → Explain → ConsentGate → Watchlist → EscalationDraft → Persist
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {PIPELINE_STEPS.map((step, i) => (
              <motion.span
                key={step}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="rounded-xl bg-muted px-3 py-2 text-sm font-medium"
              >
                {step}
              </motion.span>
            ))}
          </div>
        </CardContent>
      </Card>

      <AgentTrace steps={lastRunLogs} />

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Dry run</CardTitle>
          <p className="text-muted-foreground text-sm">
            Paste a transcript or upload a JSON event batch to preview what the pipeline would do. Never sends.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder='Paste transcript or JSON, e.g. [{"event_type": "final_asr", "payload": {"text": "..."}}]'
            value={dryRunInput}
            onChange={(e) => setDryRunInput(e.target.value)}
            className="min-h-[120px] rounded-xl font-mono text-sm"
          />
          <Button
            className="rounded-xl"
            onClick={() => setLastRunLogs(MOCK_TRACE)}
          >
            Preview pipeline
          </Button>
          <p className="text-muted-foreground text-xs">
            Outputs: risk signal preview, explanation preview, watchlist preview, escalation draft (never sent).
          </p>
        </CardContent>
      </Card>

      <PolicyGateCard
        withheld={["Raw transcript withheld until consent."]}
        couldShare={["Full pipeline trace for caregiver."]}
      />
    </div>
  );
}
