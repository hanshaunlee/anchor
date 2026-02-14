"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { ArrowLeft, Network } from "lucide-react";

type RingDetail = Awaited<ReturnType<typeof api.getRing>>;

export default function RingDetailPage({ params }: { params: { id: string } }) {
  const [ring, setRing] = useState<RingDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!params.id) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getRing(params.id);
      setRing(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load ring");
    } finally {
      setLoading(false);
    }
  }, [params.id]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  if (error || !ring) {
    return (
      <div className="space-y-4">
        <Link href="/rings">
          <Button variant="ghost" size="sm" className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Rings
          </Button>
        </Link>
        <Card className="rounded-2xl">
          <CardContent className="pt-6">
            <p className="text-destructive">{error ?? "Ring not found."}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const meta = (ring.meta ?? {}) as { member_count?: number; top_connectors?: Array<{ entity_id?: string; betweenness?: number; bridge_risk?: number }>; evidence_edge_count?: number };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/rings">
          <Button variant="ghost" size="sm" className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Rings
          </Button>
        </Link>
        <span className="text-muted-foreground text-sm">
          {ring.created_at ? new Date(ring.created_at).toLocaleString() : ""}
        </span>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Network className="h-5 w-5" />
            Ring detail
          </CardTitle>
          <p className="text-muted-foreground text-sm">
            Score {ring.score.toFixed(2)}
            {meta.member_count != null ? ` · ${meta.member_count} members` : ""}
            {meta.evidence_edge_count != null ? ` · ${meta.evidence_edge_count} evidence edges` : ""}
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {ring.members && ring.members.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">Members</p>
              <ul className="space-y-1 text-sm">
                {ring.members.map((m, i) => (
                  <li key={i} className="flex items-center gap-2">
                    <span className="font-mono text-xs truncate max-w-[200px]" title={m.entity_id ?? ""}>
                      {m.entity_id ? `${m.entity_id.slice(0, 8)}…` : "—"}
                    </span>
                    {m.role && <span className="text-muted-foreground text-xs">{m.role}</span>}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {meta.top_connectors && meta.top_connectors.length > 0 && (
            <div>
              <p className="text-xs font-medium text-muted-foreground mb-2">Top connectors</p>
              <ul className="space-y-1 text-sm">
                {meta.top_connectors.map((c, i) => (
                  <li key={i}>
                    {c.entity_id ? `${String(c.entity_id).slice(0, 8)}…` : "—"}
                    {c.betweenness != null ? ` · betweenness ${c.betweenness.toFixed(2)}` : ""}
                    {c.bridge_risk != null ? ` · bridge_risk ${c.bridge_risk.toFixed(2)}` : ""}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            Related alerts: open Alerts and filter by signal_type &quot;ring_candidate&quot; or open an alert that references this ring.
          </p>
          <Link href="/alerts">
            <Button variant="outline" size="sm">
              View alerts
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
