"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RingCard } from "@/components/protection/RingCard";
import { useProtectionRings, useRings } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import { Network, ChevronRight } from "lucide-react";
import type { ProtectionRingSummary } from "@/lib/api/schemas";

function legacyToRingSummary(ring: { id: string; household_id: string; created_at: string; updated_at: string; score: number; meta: Record<string, unknown> }): ProtectionRingSummary {
  const meta = ring.meta ?? {};
  const memberCount = (meta.member_count as number) ?? 0;
  return {
    id: ring.id,
    household_id: ring.household_id,
    created_at: ring.created_at,
    updated_at: ring.updated_at,
    score: ring.score,
    summary_label: (meta.summary_label as string) ?? null,
    summary_text: (meta.summary_text as string) ?? null,
    members_count: memberCount,
    meta,
  };
}

function dedupeRingsForDisplay(rings: ProtectionRingSummary[]): { ring: ProtectionRingSummary; similarCount: number }[] {
  const key = (r: ProtectionRingSummary) => {
    const label = r.summary_label || (r.meta?.topics as string[] | undefined)?.slice(0, 2).join("+") || "";
    return `${label}|${r.members_count}`;
  };
  const byKey = new Map<string, ProtectionRingSummary[]>();
  for (const r of rings) {
    const k = key(r);
    if (!byKey.has(k)) byKey.set(k, []);
    byKey.get(k)!.push(r);
  }
  return Array.from(byKey.entries()).map(([, group]) => ({
    ring: group[0],
    similarCount: group.length - 1,
  }));
}

export default function RingsPage() {
  const demoMode = useAppStore((s) => s.demoMode);
  const { data: protectionRings, isLoading: protectionLoading } = useProtectionRings();
  const { data: legacyData, isLoading: legacyLoading } = useRings();

  const rings: ProtectionRingSummary[] =
    protectionRings && protectionRings.length > 0
      ? protectionRings
      : (legacyData?.rings ?? []).map(legacyToRingSummary);
  const ringsDeduped = useMemo(() => dedupeRingsForDisplay(rings), [rings]);
  const useLegacy = !protectionRings || protectionRings.length === 0;
  const isLoading = protectionLoading || (useLegacy && legacyLoading);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Connected patterns</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Possible scam networks we&apos;ve detected. Each pattern shows why it matters and what to do.
        </p>
        <Link href="/protection" className="text-xs text-primary hover:underline mt-1 inline-flex items-center gap-0.5">
          View on Protection
          <ChevronRight className="h-3 w-3" />
        </Link>
      </div>

      <Card className="rounded-2xl border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Network className="h-4 w-4" />
            Discovered patterns
          </CardTitle>
          <p className="text-muted-foreground text-sm">
            Risk level, &quot;why it matters,&quot; and top entities. Open a card for plain-English explanation and what to do next.
          </p>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-20 w-full rounded-xl" />
              ))}
            </div>
          ) : rings.length === 0 ? (
            <p className="text-muted-foreground py-12 text-center">
              No connected patterns yet. Run an investigation from the Automation page when the graph has enough connections.
            </p>
          ) : (
            <ul className="space-y-3">
              {ringsDeduped.map(({ ring, similarCount }) => (
                <li key={ring.id}>
                  <RingCard ring={ring} showChangeBadge similarCount={similarCount} />
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
