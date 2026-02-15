"use client";

import { useRef, useMemo, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useProtectionOverview, useHouseholdMe } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Shield,
  RefreshCw,
  CircleDot,
  FileText,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  WatchlistSectionCard,
  groupWatchlistItems,
} from "@/components/protection/WatchlistSectionCard";
import { RingCard } from "@/components/protection/RingCard";
import { LastInvestigationStrip } from "@/components/protection/LastInvestigationStrip";

const SECTION_ORDER = ["device_policy", "contact", "phrase", "topic"] as const;

function formatRelative(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffM = Math.floor(diffMs / 60_000);
  const diffH = Math.floor(diffMs / 3_600_000);
  const diffD = Math.floor(diffMs / 86_400_000);
  if (diffM < 1) return "just now";
  if (diffM < 60) return `${diffM}m ago`;
  if (diffH < 24) return `${diffH}h ago`;
  if (diffD < 7) return `${diffD}d ago`;
  return d.toLocaleDateString();
}

export default function ProtectionPage() {
  const demoMode = useAppStore((s) => s.demoMode);
  const [showDuplicates, setShowDuplicates] = useState(false);
  const [showAgentProvenance, setShowAgentProvenance] = useState(false);
  const { data: me } = useHouseholdMe();
  const isAdmin = me?.role === "admin";
  const { data: overview, isLoading, refetch, isFetching } = useProtectionOverview();
  const refreshRef = useRef<HTMLButtonElement>(null);

  const watchlistBySection = useMemo(() => {
    if (!overview?.watchlist_summary.items.length) return new Map<string, typeof overview.watchlist_summary.items>();
    return groupWatchlistItems(overview.watchlist_summary.items, !showDuplicates);
  }, [overview?.watchlist_summary.items, showDuplicates]);

  return (
    <div className="space-y-6">
      {demoMode && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-800 dark:text-amber-200">
          <span className="font-medium">Demo Data Mode</span> – Using fixture data.
        </div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Protection</h1>
          <p className="text-muted-foreground text-sm mt-1">
            What Anchor is watching right now, and what clusters we&apos;ve detected.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={demoMode ? "secondary" : "default"} className="font-mono text-xs">
            {demoMode ? "Demo" : "Live API"}
          </Badge>
          {overview?.last_updated_at && (
            <span className="text-xs text-muted-foreground">
              Updated {formatRelative(overview.last_updated_at)}
            </span>
          )}
          <Button
            ref={refreshRef}
            variant="outline"
            size="sm"
            className="rounded-xl"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw className={cn("h-4 w-4 mr-1.5", isFetching && "animate-spin")} />
            Refresh
          </Button>
        </div>
      </div>

      <LastInvestigationStrip />

      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full rounded-2xl" />
          <Skeleton className="h-64 w-full rounded-2xl" />
        </div>
      ) : overview ? (
        <>
          {/* Overview cards */}
          <div className="grid gap-4 sm:grid-cols-3">
            <Card className="rounded-2xl border-border">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Shield className="h-4 w-4" />
                  Watchlists
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-semibold tabular-nums">{overview.watchlist_summary.total}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Active items · {Object.keys(overview.watchlist_summary.by_category).length} categories
                </p>
              </CardContent>
            </Card>
            <Card className="rounded-2xl border-border">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <CircleDot className="h-4 w-4" />
                  Rings
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-semibold tabular-nums">{overview.rings_summary.length}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Active clusters
                </p>
              </CardContent>
            </Card>
            <Card className="rounded-2xl border-border">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Reports
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-semibold tabular-nums">
                  {overview.reports_summary.filter((r) => r.last_run_at).length}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Model health · Calibration · Narrative
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            {/* Watchlists: 4 grouped sections with chips */}
            <div className="lg:col-span-2 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-base font-semibold">Watchlists</h2>
                <div className="flex items-center gap-4">
                  {isAdmin && (
                    <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showAgentProvenance}
                        onChange={(e) => setShowAgentProvenance(e.target.checked)}
                        className="rounded border-border"
                      />
                      Show agent provenance
                    </label>
                  )}
                  <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showDuplicates}
                      onChange={(e) => setShowDuplicates(e.target.checked)}
                      className="rounded border-border"
                    />
                    Show duplicates (debug)
                  </label>
                </div>
              </div>
              {overview.watchlist_summary.items.length === 0 ? (
                <Card className="rounded-2xl border-border">
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    No active watchlist items.
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-4 sm:grid-cols-2">
                  {SECTION_ORDER.map((section) => {
                    const items = watchlistBySection.get(section) ?? [];
                    const why = items[0]?.explanation ?? undefined;
                    return (
                      <WatchlistSectionCard
                        key={section}
                        section={section}
                        items={items}
                        topN={6}
                        why={why}
                        lastUpdated={overview.last_updated_at ?? undefined}
                        showTypeConfidence
                        showAgentProvenance={showAgentProvenance}
                        viewAllHref="/watchlists"
                        viewAllLabel={`View all ${items.length}`}
                      />
                    );
                  })}
                </div>
              )}
            </div>

            {/* Rings + Reports sidebar */}
            <div className="space-y-6">
              <Card className="rounded-2xl border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Rings</CardTitle>
                  <p className="text-muted-foreground text-xs">Clusters of connected entities.</p>
                </CardHeader>
                <CardContent>
                  {overview.rings_summary.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No rings detected.</p>
                  ) : (
                    <ul className="space-y-2">
                      {overview.rings_summary.slice(0, 5).map((ring) => (
                        <li key={ring.id}>
                          <RingCard ring={ring} />
                        </li>
                      ))}
                    </ul>
                  )}
                  {overview.rings_summary.length > 0 && (
                    <Link href="/rings" className="block mt-2">
                      <Button variant="ghost" size="sm" className="w-full rounded-xl text-xs">
                        View all rings
                      </Button>
                    </Link>
                  )}
                </CardContent>
              </Card>

              <Card className="rounded-2xl border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">Reports</CardTitle>
                  <p className="text-muted-foreground text-xs">Model health, calibration, redteam, narrative.</p>
                </CardHeader>
                <CardContent className="space-y-2">
                  {overview.reports_summary.map((r) => (
                    <div
                      key={r.kind}
                      className="flex items-center justify-between rounded-lg border border-border bg-muted/10 px-3 py-2"
                    >
                      <span className="text-sm font-medium">{r.kind}</span>
                      {r.last_run_at ? (
                        <span className="text-xs text-muted-foreground">
                          {new Date(r.last_run_at).toLocaleDateString()}
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </div>
                  ))}
                  <Link href="/reports">
                    <Button variant="outline" size="sm" className="w-full mt-2 rounded-xl">
                      Open full reports
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </div>
          </div>
        </>
      ) : (
        <Card className="rounded-2xl border-dashed border-border p-8 text-center text-muted-foreground">
          <AlertCircle className="h-10 w-10 mx-auto mb-2 opacity-50" />
          <p>Unable to load protection data.</p>
        </Card>
      )}
    </div>
  );
}
