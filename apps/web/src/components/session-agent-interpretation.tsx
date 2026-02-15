"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Network, Layers, Shield, GitBranch } from "lucide-react";

export type SessionAgentInterpretationProps = {
  /** Entity IDs or labels from agent output (e.g. linked risk signal). */
  entities?: { id: string; label?: string; type?: string }[];
  /** Structural motifs (e.g. triadic, star). */
  motifs?: string[];
  /** Independence cluster or rule ID. */
  independenceClusterId?: string | null;
  /** Graph edges created (count or list). */
  graphEdgesCount?: number;
  /** Risk score contribution (0–1 or percentage). */
  riskScoreContribution?: number | null;
  /** When no agent data is available for this session. */
  emptyMessage?: string;
  className?: string;
};

export function SessionAgentInterpretation({
  entities = [],
  motifs = [],
  independenceClusterId,
  graphEdgesCount,
  riskScoreContribution,
  emptyMessage = "No agent interpretation for this session alone. Open a linked alert to see entities, motifs, and graph updates.",
  className,
}: SessionAgentInterpretationProps) {
  const hasData =
    entities.length > 0 ||
    motifs.length > 0 ||
    independenceClusterId != null ||
    graphEdgesCount != null ||
    (riskScoreContribution != null && riskScoreContribution > 0);

  return (
    <Card className={cn("rounded-2xl shadow-sm border-border", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Network className="h-4 w-4" />
          Agent interpretation
        </CardTitle>
        <p className="text-muted-foreground text-xs">
          Entities extracted, structural motifs, graph edges, and risk contribution from pipeline.
        </p>
      </CardHeader>
      <CardContent>
        {!hasData ? (
          <p className="text-sm text-muted-foreground py-4">{emptyMessage}</p>
        ) : (
          <div className="space-y-4">
            {entities.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1">
                  <Layers className="h-3.5 w-3" />
                  Entities extracted
                </p>
                <ul className="flex flex-wrap gap-1.5">
                  {entities.slice(0, 20).map((e) => (
                    <li
                      key={e.id}
                      className="rounded-md bg-muted/50 px-2 py-1 font-mono text-xs"
                      title={e.type}
                    >
                      {e.label ?? e.id}
                    </li>
                  ))}
                  {entities.length > 20 && (
                    <li className="text-xs text-muted-foreground">
                      +{entities.length - 20} more
                    </li>
                  )}
                </ul>
              </div>
            )}
            {motifs.length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">
                  Structural motifs
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {motifs.map((m, i) => (
                    <span
                      key={i}
                      className="rounded-md border border-border bg-card px-2 py-1 text-xs"
                    >
                      {m}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {independenceClusterId != null && independenceClusterId !== "" && (
              <div className="flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Independence cluster
                  </p>
                  <p className="font-mono text-xs">{independenceClusterId}</p>
                </div>
              </div>
            )}
            {graphEdgesCount != null && (
              <div className="flex items-center gap-2">
                <Network className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Graph edges created
                  </p>
                  <p className="font-mono text-xs">{graphEdgesCount}</p>
                </div>
              </div>
            )}
            {riskScoreContribution != null && riskScoreContribution > 0 && (
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-muted-foreground" />
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Risk score contribution
                  </p>
                  <p className="font-mono text-xs">
                    {(riskScoreContribution * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
