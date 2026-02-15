"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle2, AlertCircle, Minus } from "lucide-react";
import { TraceViewer } from "@/components/trace-viewer";
import { cn } from "@/lib/utils";

/** Map pipeline steps to capability chips: Detect, Explain, Connect, Validate, Attack test */
const CAPABILITIES: { id: string; label: string; steps: string[] }[] = [
  { id: "detect", label: "Detect (financial detection)", steps: ["Ingest", "GraphUpdate", "Score"] },
  { id: "explain", label: "Explain (narratives)", steps: ["Explain"] },
  { id: "connect", label: "Connect (ring/pattern discovery)", steps: ["Watchlist"] },
  { id: "validate", label: "Validate (health/calibration/drift)", steps: ["ConsentGate"] },
  { id: "attack", label: "Attack test (red-team)", steps: [] },
];

function stepStatus(
  stepTrace: Array<{ step?: string; status?: string; error?: string }>,
  steps: string[]
): "ok" | "warn" | "none" {
  if (steps.length === 0) return "none";
  const relevant = stepTrace.filter((s) => steps.includes(s.step ?? ""));
  if (relevant.length === 0) return "none";
  const hasFail = relevant.some((s) => (s.status ?? "").toLowerCase() === "error" || (s.status ?? "").toLowerCase() === "fail");
  const hasSuccess = relevant.some((s) => (s.status ?? "").toLowerCase() === "success" || (s.status ?? "").toLowerCase() === "ok");
  if (hasFail) return "warn";
  if (hasSuccess) return "ok";
  return "none";
}

export type AgentStripProps = {
  stepTrace: Array<{ step?: string; status?: string; error?: string; notes?: string }>;
  runId?: string | null;
  timestamp?: string | null;
};

export function AgentStrip({ stepTrace, runId, timestamp }: AgentStripProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = CAPABILITIES.find((c) => c.id === selectedId);
  const relevantSteps = selected
    ? stepTrace.filter((s) => selected.steps.includes(s.step ?? ""))
    : [];

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Agents involved</CardTitle>
        <p className="text-muted-foreground text-sm">
          Pipeline capabilities. Click a chip for last run summary and trace.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {CAPABILITIES.map((cap) => {
            const status = stepStatus(stepTrace, cap.steps);
            const isSelected = selectedId === cap.id;
            return (
              <button
                key={cap.id}
                type="button"
                onClick={() => setSelectedId(isSelected ? null : cap.id)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-sm font-medium transition",
                  isSelected
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border bg-muted/30 hover:bg-muted/50 text-foreground"
                )}
              >
                {status === "ok" && <CheckCircle2 className="h-4 w-4 text-green-600" />}
                {status === "warn" && <AlertCircle className="h-4 w-4 text-amber-600" />}
                {status === "none" && <Minus className="h-4 w-4 text-muted-foreground" />}
                {cap.label}
              </button>
            );
          })}
        </div>

        {selected && (
          <div className="rounded-xl border border-border bg-muted/20 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <p className="font-medium text-sm">{selected.label}</p>
              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setSelectedId(null)}>
                Close
              </Button>
            </div>
            {(runId || timestamp) && (
              <p className="text-xs text-muted-foreground">
                {timestamp && new Date(timestamp).toLocaleString()}
                {runId && ` · Run: ${runId.slice(0, 8)}…`}
              </p>
            )}
            {relevantSteps.length > 0 ? (
              <TraceViewer stepTrace={relevantSteps} title="Steps in this run" className="!shadow-none !border-0 !p-0" />
            ) : (
              <p className="text-sm text-muted-foreground">
                No steps in this run for this capability. Run a full Safety Check to see trace.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
