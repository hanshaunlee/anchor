"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

export type WhatChangedPanelProps = {
  /** From explanation.what_changed_summary or backend */
  whatChangedSummary?: string | null;
  /** Ring / graph metadata: e.g. "47 entities connected in a tight cluster" */
  ringOrGraphSummary?: string | null;
  /** If true, show "no changes" message */
  hasNoChanges?: boolean;
};

export function WhatChangedPanel({
  whatChangedSummary,
  ringOrGraphSummary,
  hasNoChanges,
}: WhatChangedPanelProps) {
  const message =
    whatChangedSummary?.trim() ||
    (hasNoChanges ? "No new structural changes in the last 7 days." : null) ||
    ringOrGraphSummary?.trim() ||
    "No structural change summary available for this signal.";

  return (
    <Card className="rounded-2xl shadow-sm border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <TrendingUp className="h-4 w-4" />
          What changed in the household graph?
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground leading-relaxed">{message}</p>
      </CardContent>
    </Card>
  );
}
