"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ExplainableIds } from "@/components/explainable-ids";
import { useState } from "react";

export type ActionRow = {
  id: string;
  status?: string;
  channel?: string | null;
  triggered_by_risk_signal_id?: string | null;
  provider_message_id?: string | null;
  error?: string | null;
  created_at?: string | null;
  sent_at?: string | null;
  recipient_contact_last4?: string | null;
};

export type ActionsHistoryProps = {
  actions: ActionRow[];
  /** Filter by status (sent/failed/pending/queued/delivered) */
  statusFilter?: string;
  onStatusFilterChange?: (status: string) => void;
};

const STATUS_OPTIONS = ["all", "sent", "delivered", "queued", "failed", "suppressed"];

export function ActionsHistory({
  actions,
  statusFilter: statusFilterProp = "all",
  onStatusFilterChange,
}: ActionsHistoryProps) {
  const [statusFilterLocal, setStatusFilterLocal] = useState<string>("all");
  const statusFilter = onStatusFilterChange ? statusFilterProp : statusFilterLocal;
  const setStatusFilter = onStatusFilterChange ?? setStatusFilterLocal;
  const [channelFilter, setChannelFilter] = useState<string>("all");
  const [alertFilter, setAlertFilter] = useState<string>("");

  const filtered = actions.filter((a) => {
    if (statusFilter !== "all" && (a.status ?? "").toLowerCase() !== statusFilter) return false;
    if (channelFilter !== "all" && (a.channel ?? "") !== channelFilter) return false;
    if (alertFilter && !(a.triggered_by_risk_signal_id ?? "").toLowerCase().includes(alertFilter.toLowerCase())) return false;
    return true;
  });

  const channels = [...new Set(actions.map((a) => a.channel).filter(Boolean))] as string[];

  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Outreach history</CardTitle>
        <p className="text-muted-foreground text-sm">
          Filter by status, channel, or alert ID. Provider receipt shown when present.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-sm text-muted-foreground">Status:</span>
          {STATUS_OPTIONS.map((s) => (
            <Button
              key={s}
              variant={statusFilter === s ? "default" : "outline"}
              size="sm"
              className="rounded-lg h-8"
              onClick={() => setStatusFilter(s)}
            >
              {s}
            </Button>
          ))}
          {channels.length > 0 && (
            <>
              <span className="text-sm text-muted-foreground ml-2">Channel:</span>
              <select
                className="rounded-lg border border-input bg-background px-2 py-1 text-sm h-8"
                value={channelFilter}
                onChange={(e) => setChannelFilter(e.target.value)}
              >
                <option value="all">All</option>
                {channels.map((ch) => (
                  <option key={ch} value={ch}>{ch}</option>
                ))}
              </select>
            </>
          )}
          <input
            type="text"
            placeholder="Alert ID"
            className="rounded-lg border border-input bg-background px-2 py-1 text-sm w-40 h-8"
            value={alertFilter}
            onChange={(e) => setAlertFilter(e.target.value)}
          />
        </div>
        <div className="rounded-lg border border-border divide-y divide-border">
          {filtered.length === 0 && (
            <div className="text-muted-foreground text-sm text-center py-8 px-4">
              No actions match filters.
            </div>
          )}
          {filtered.slice(0, 50).map((a) => (
            <div key={a.id} className="flex flex-wrap items-center gap-4 px-4 py-3 text-sm">
              <Badge variant={a.status === "sent" || a.status === "delivered" ? "default" : a.status === "failed" ? "destructive" : "secondary"} className="rounded shrink-0">
                {a.status ?? "—"}
              </Badge>
              {a.triggered_by_risk_signal_id ? (
                <Link href={`/alerts/${a.triggered_by_risk_signal_id}`}>
                  <Button variant="link" className="h-auto p-0 text-xs">
                    Alert {String(a.triggered_by_risk_signal_id).slice(0, 8)}…
                  </Button>
                </Link>
              ) : <span className="text-muted-foreground">—</span>}
              <span className="text-muted-foreground shrink-0">{a.channel ?? "—"}</span>
              <span className="text-muted-foreground shrink-0">
                {a.created_at ? new Date(a.created_at).toLocaleString() : "—"}
              </span>
              <span className="text-xs text-muted-foreground truncate max-w-[140px]" title={a.provider_message_id ?? a.error ?? undefined}>
                {a.provider_message_id ?? a.error ?? "—"}
              </span>
            </div>
          ))}
        </div>
        {filtered.some((a) => a.triggered_by_risk_signal_id) && (
          <ExplainableIds
            context="alert_ids"
            items={[...new Set(filtered.map((a) => a.triggered_by_risk_signal_id).filter(Boolean))].slice(0, 15).map((id) => ({ id: String(id), label: null }))}
            title="What these alerts are"
            className="mt-4 pt-4 border-t border-border"
          />
        )}
      </CardContent>
    </Card>
  );
}
