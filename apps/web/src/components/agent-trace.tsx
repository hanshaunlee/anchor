"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { CheckCircle2, AlertCircle, XCircle, Clock } from "lucide-react";

export type TraceStep = {
  step: string;
  description: string;
  inputs?: string;
  outputs?: string;
  status: "success" | "warn" | "fail" | "pending";
  latency_ms?: number;
};

export function AgentTrace({ steps, className, title = "Agent trace" }: { steps: TraceStep[]; className?: string; title?: string }) {
  return (
    <Card className={cn("rounded-2xl shadow-sm", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{title}</CardTitle>
        <p className="text-muted-foreground text-sm">
          Pipeline steps in plain English. Not developer logs.
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {steps.map((s, i) => (
          <div
            key={i}
            className="rounded-xl border border-border bg-muted/30 p-4 transition hover:bg-muted/50"
          >
            <div className="flex items-start gap-3">
              <span className="shrink-0">
                {s.status === "success" && (
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                )}
                {s.status === "warn" && (
                  <AlertCircle className="h-5 w-5 text-amber-600" />
                )}
                {s.status === "fail" && (
                  <XCircle className="h-5 w-5 text-destructive" />
                )}
                {(s.status === "pending" || !["success", "warn", "fail"].includes(s.status)) && (
                  <Clock className="h-5 w-5 text-muted-foreground" />
                )}
              </span>
              <div className={cn("min-w-0 flex-1 space-y-1", s.status === "pending" && "opacity-70")}>
                <p className={cn("font-medium text-sm", s.status === "pending" && "text-muted-foreground")}>{s.step}</p>
                <p className="text-muted-foreground text-sm">{s.description}</p>
                {(s.inputs || s.outputs) && (
                  <div className="text-muted-foreground text-xs flex flex-wrap gap-2 mt-2">
                    {s.inputs && (
                      <span>
                        <strong>In:</strong> {s.inputs}
                      </span>
                    )}
                    {s.outputs && (
                      <span>
                        <strong>Out:</strong> {s.outputs}
                      </span>
                    )}
                  </div>
                )}
                {s.latency_ms != null && (
                  <p className="text-muted-foreground text-xs mt-1">
                    {s.latency_ms} ms
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
