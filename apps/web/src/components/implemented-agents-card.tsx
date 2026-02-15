"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle2, AlertCircle, Minus, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

type CatalogEntry = {
  agent_name: string;
  slug: string;
  label: string;
  display_name?: string;
  tier?: string;
  runnable?: boolean;
  reason?: string | null;
};

type StatusAgent = {
  agent_name: string;
  last_run_id?: string | null;
  last_run_at?: string | null;
  last_run_status?: string | null;
};

const TIER_ORDER = ["INVESTIGATION", "USER_ACTION", "SYSTEM_MAINTENANCE", "DEVTOOLS"] as const;
const TIER_LABELS: Record<string, string> = {
  INVESTIGATION: "Investigation",
  USER_ACTION: "User action",
  SYSTEM_MAINTENANCE: "System maintenance",
  DEVTOOLS: "Dev / safety testing",
};

function statusIcon(status: string | null | undefined): "ok" | "warn" | "none" {
  const s = (status ?? "").toLowerCase();
  if (s === "success" || s === "completed" || s === "ok") return "ok";
  if (s === "error" || s === "fail") return "warn";
  return "none";
}

export type ImplementedAgentsCardProps = {
  catalog: CatalogEntry[];
  statusAgents?: StatusAgent[];
};

export function ImplementedAgentsCard({ catalog, statusAgents = [] }: ImplementedAgentsCardProps) {
  const byName = new Map(statusAgents.map((a) => [a.agent_name, a]));

  const byTier = new Map<string, CatalogEntry[]>();
  for (const entry of catalog) {
    const tier = entry.tier ?? "OTHER";
    if (!byTier.has(tier)) byTier.set(tier, []);
    byTier.get(tier)!.push(entry);
  }

  const orderedTiers = TIER_ORDER.filter((t) => byTier.has(t));

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Cpu className="h-4 w-4" />
          Implemented agents
        </CardTitle>
        <p className="text-muted-foreground text-sm">
          All agents in the system. Last run status from your household.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {orderedTiers.length === 0 ? (
          <p className="text-sm text-muted-foreground">No agent catalog loaded.</p>
        ) : (
          orderedTiers.map((tier) => {
            const entries = byTier.get(tier) ?? [];
            return (
              <div key={tier}>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                  {TIER_LABELS[tier] ?? tier}
                </h4>
                <ul className="space-y-1.5">
                  {entries.map((entry) => {
                    const status = byName.get(entry.agent_name);
                    const icon = statusIcon(status?.last_run_status);
                    return (
                      <li
                        key={entry.agent_name}
                        className={cn(
                          "flex items-center gap-2 rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm"
                        )}
                      >
                        {icon === "ok" && <CheckCircle2 className="h-4 w-4 shrink-0 text-green-600" />}
                        {icon === "warn" && <AlertCircle className="h-4 w-4 shrink-0 text-amber-600" />}
                        {icon === "none" && <Minus className="h-4 w-4 shrink-0 text-muted-foreground" />}
                        <span className="font-medium min-w-0 truncate">
                          {entry.label}
                        </span>
                        {entry.display_name && entry.display_name !== entry.label && (
                          <span className="text-muted-foreground text-xs truncate hidden sm:inline">
                            {entry.display_name}
                          </span>
                        )}
                        {status?.last_run_at && (
                          <span className="text-xs text-muted-foreground ml-auto shrink-0">
                            {new Date(status.last_run_at).toLocaleDateString()}
                          </span>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}
