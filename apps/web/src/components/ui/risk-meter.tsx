"use client";

import { cn } from "@/lib/utils";

const SEVERITY_COLORS: Record<number, string> = {
  1: "bg-severity-1",
  2: "bg-severity-2",
  3: "bg-severity-3",
  4: "bg-severity-4",
  5: "bg-severity-5",
};

/** Risk confidence meter: progress bar + optional decision-rule caption. */
export function RiskMeter({
  score,
  severity = 1,
  decisionRuleUsed,
  caption,
  className,
}: {
  score: number;
  severity?: number;
  decisionRuleUsed?: string | null;
  caption?: string | null;
  className?: string;
}) {
  const pct = Math.round(score * 100);
  const barColor = SEVERITY_COLORS[severity] ?? "bg-primary";

  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">Risk confidence</span>
        <span className="text-sm font-medium tabular-nums">{pct}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={cn("h-full rounded-full transition-all", barColor)}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
      {(caption || decisionRuleUsed) && (
        <p className="text-xs text-muted-foreground">
          {caption ?? (decisionRuleUsed ? `Decision rule: ${decisionRuleUsed}` : null)}
        </p>
      )}
    </div>
  );
}
