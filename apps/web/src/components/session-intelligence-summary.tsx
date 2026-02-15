"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { SessionListItem } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";
import { FileText, Shield, AlertCircle } from "lucide-react";

export type SessionIntelligenceSummaryProps = {
  session: SessionListItem;
  /** When this session is linked to risk signals (e.g. from alert page). */
  riskSignalsCount?: number;
  /** Entities extracted (from linked signal or backend). */
  entitiesCount?: number;
  /** Topics detected (optional). */
  topics?: string[];
  /** Outbound eligibility from household consent. */
  outboundEligible?: boolean;
  /** Model was run for this session / linked signal. */
  modelAvailable?: boolean;
  className?: string;
};

export function SessionIntelligenceSummary({
  session,
  riskSignalsCount = 0,
  entitiesCount,
  topics = [],
  outboundEligible = false,
  modelAvailable,
  className,
}: SessionIntelligenceSummaryProps) {
  const hasConsent = session.consent_state && Object.keys(session.consent_state).length > 0;
  const narrative = session.summary_text ?? "No summary generated.";

  return (
    <Card className={cn("rounded-2xl shadow-sm border-border", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Session intelligence overview
        </CardTitle>
        <p className="text-muted-foreground text-xs">One-line narrative, risk signals, and consent.</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Narrative</p>
          <p className="text-sm text-foreground">{narrative}</p>
        </div>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-muted-foreground shrink-0" />
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Risk signals</p>
              <p className="font-mono font-medium">
                {riskSignalsCount > 0 ? riskSignalsCount : "None"}
              </p>
            </div>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Entities extracted</p>
            <p className="font-mono font-medium">{entitiesCount != null ? entitiesCount : "—"}</p>
          </div>
          {topics.length > 0 && (
            <div className="col-span-2">
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Topics</p>
              <p className="text-sm">{topics.join(", ")}</p>
            </div>
          )}
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-muted-foreground shrink-0" />
            <div>
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Outbound eligibility</p>
              <p className={outboundEligible ? "text-emerald-600 dark:text-emerald-400 font-medium" : "text-muted-foreground"}>
                {outboundEligible ? "Eligible" : "Off"}
              </p>
            </div>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Consent / redaction</p>
            <p className={hasConsent ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground"}>
              {hasConsent ? "Set (redaction applied)" : "Default"}
            </p>
          </div>
        </div>
        {modelAvailable === false && (
          <div className="rounded-lg bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
            Model not run for this session; metrics show rule/motif only where available.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
