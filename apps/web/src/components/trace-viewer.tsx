"use client";

import { AgentTrace, type TraceStep } from "@/components/agent-trace";

function stepTraceToTraceSteps(
  stepTrace: Array<{ step?: string; status?: string; error?: string; notes?: string; outputs_count?: number; started_at?: string; ended_at?: string }>
): TraceStep[] {
  if (!Array.isArray(stepTrace)) return [];
  return stepTrace.map((item) => {
    const step = (item.step ?? "step") as string;
    const desc = item.error ?? item.notes ?? (item.status === "ok" ? "Completed" : item.status ?? "pending");
    const status: "success" | "warn" | "fail" =
      item.status === "ok" || item.status === "success"
        ? "success"
        : item.status === "error" || item.status === "fail"
          ? "fail"
          : "warn";
    const outputs =
      item.outputs_count != null ? `outputs: ${item.outputs_count}` : undefined;
    return { step, description: String(desc), status, outputs };
  });
}

type TraceViewerProps = {
  stepTrace: Array<{ step?: string; status?: string; error?: string; notes?: string; outputs_count?: number }>;
  title?: string;
  className?: string;
};

export function TraceViewer({ stepTrace, title = "Run trace", className }: TraceViewerProps) {
  const steps = stepTraceToTraceSteps(stepTrace);
  if (steps.length === 0) return null;
  return (
    <AgentTrace steps={steps} className={className} title={title} />
  );
}
