"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AgentTrace, type TraceStep } from "@/components/agent-trace";
import { ChevronDown, ChevronRight, FlaskConical } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SimilarIncidentsResponse } from "@/lib/api/schemas";

export type AnalysisDetailsCollapsibleProps = {
  /** Similar incidents from page */
  similarData: SimilarIncidentsResponse | null;
  /** Agent trace steps (from explanation.logs or default) */
  traceSteps: TraceStep[];
  /** Judge mode: show raw model internals */
  judgeMode: boolean;
  /** When judge mode: raw explanation object for JSON dump */
  explanationJson?: Record<string, unknown> | null;
  /** When judge mode: decision_rule_used, calibrated_p, rule_score, fusion_score */
  decisionRuleUsed?: string | null;
  calibratedP?: number | null;
  ruleScore?: number | null;
  fusionScore?: number | null;
  structuralMotifsJson?: unknown;
  /** Default open state */
  defaultOpen?: boolean;
};

export function AnalysisDetailsCollapsible({
  similarData,
  traceSteps,
  judgeMode,
  explanationJson,
  decisionRuleUsed,
  calibratedP,
  ruleScore,
  fusionScore,
  structuralMotifsJson,
  defaultOpen = false,
}: AnalysisDetailsCollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Card className="rounded-2xl shadow-sm border-border">
      <button
        type="button"
        className="w-full flex items-center gap-2 p-5 text-left hover:bg-muted/30 rounded-t-2xl transition"
        onClick={() => setOpen(!open)}
      >
        <FlaskConical className="h-5 w-5 text-muted-foreground" />
        <span className="font-semibold text-lg">Analysis & model details</span>
        {open ? (
          <ChevronDown className="h-4 w-4 ml-auto text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 ml-auto text-muted-foreground" />
        )}
      </button>
      {open && (
        <CardContent className="pt-0 space-y-6">
          {/* Similar incidents */}
          <section>
            <CardTitle className="text-base mb-2">Similar incidents</CardTitle>
            {similarData?.available !== true ? (
              <p className="text-muted-foreground text-sm">
                {similarData?.reason === "model_not_run"
                  ? "Unavailable (model not run)."
                  : "Similar incidents require GNN embeddings; none stored for this signal."}
              </p>
            ) : similarData?.similar && similarData.similar.length > 0 ? (
              <>
                <ul className="space-y-2">
                  {similarData.similar.map((s) => (
                    <li
                      key={s.risk_signal_id}
                      className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border px-3 py-2 text-sm"
                    >
                      <Link href={`/alerts/${s.risk_signal_id}`} className="hover:underline font-medium">
                        {String(s.risk_signal_id).slice(0, 8)}…
                      </Link>
                      <span className="text-muted-foreground">
                        similarity {((s.similarity ?? s.score) * 100).toFixed(0)}%
                        {(s.label_outcome ?? s.outcome) && ` · ${s.label_outcome ?? s.outcome}`}
                        {s.severity != null && ` · severity ${s.severity}`}
                      </span>
                    </li>
                  ))}
                </ul>
                {similarData.retrieval_provenance && (
                  <p className="text-muted-foreground text-xs mt-3 pt-3 border-t border-border">
                    Retrieval:{" "}
                    {[
                      similarData.retrieval_provenance.model_name,
                      similarData.retrieval_provenance.embedding_dim != null &&
                        `dim ${similarData.retrieval_provenance.embedding_dim}`,
                      similarData.retrieval_provenance.timestamp &&
                        new Date(similarData.retrieval_provenance.timestamp).toLocaleString(),
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </p>
                )}
              </>
            ) : (
              <p className="text-muted-foreground text-sm">No similar past incidents.</p>
            )}
          </section>

          {/* Judge mode: model internals */}
          {judgeMode && (
            <section className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
              <h4 className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-2">
                Judge mode — internals
              </h4>
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
                {decisionRuleUsed != null && (
                  <>
                    <dt className="text-muted-foreground">decision_rule_used</dt>
                    <dd>{String(decisionRuleUsed)}</dd>
                  </>
                )}
                {calibratedP != null && (
                  <>
                    <dt className="text-muted-foreground">calibrated_p</dt>
                    <dd>{Number(calibratedP)}</dd>
                  </>
                )}
                {ruleScore != null && (
                  <>
                    <dt className="text-muted-foreground">rule_score</dt>
                    <dd>{Number(ruleScore)}</dd>
                  </>
                )}
                {fusionScore != null && (
                  <>
                    <dt className="text-muted-foreground">fusion_score</dt>
                    <dd>{Number(fusionScore)}</dd>
                  </>
                )}
              </dl>
              {structuralMotifsJson != null && (
                <div className="mt-2">
                  <p className="text-muted-foreground text-xs mb-1">structural_motifs</p>
                  <pre className="rounded-lg bg-muted/50 p-2 text-xs overflow-auto max-h-32">
                    {JSON.stringify(structuralMotifsJson, null, 2)}
                  </pre>
                </div>
              )}
              {explanationJson && (
                <div className="mt-2">
                  <p className="text-muted-foreground text-xs mb-1">explanation (full)</p>
                  <pre className="rounded-lg bg-muted/50 p-2 text-xs overflow-auto max-h-48">
                    {JSON.stringify(explanationJson, null, 2)}
                  </pre>
                </div>
              )}
            </section>
          )}

          {/* Agent trace */}
          <AgentTrace steps={traceSteps} title="Agent trace" />
        </CardContent>
      )}
    </Card>
  );
}
