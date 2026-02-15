"use client";

import { useRef, useMemo, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useProtectionOverview, useHouseholdMe, useRiskSignals } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Shield,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  WatchlistSectionCard,
  groupWatchlistItems,
} from "@/components/protection/WatchlistSectionCard";
import { RingCard } from "@/components/protection/RingCard";
import { LastInvestigationStrip } from "@/components/protection/LastInvestigationStrip";
import type { ProtectionRingSummary } from "@/lib/api/schemas";

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

/** Group rings for display: one card per "pattern" (label + size), with +N similar when duplicates. */
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

export default function ProtectionPage() {
  const demoMode = useAppStore((s) => s.demoMode);
  const [showDuplicates, setShowDuplicates] = useState(false);
  const [showAgentProvenance, setShowAgentProvenance] = useState(false);
  const [showPatterns, setShowPatterns] = useState(false);
  const { data: me } = useHouseholdMe();
  const isAdmin = me?.role === "admin";
  const { data: overview, isLoading, refetch, isFetching } = useProtectionOverview();
  const { data: signalsData } = useRiskSignals({ status: "open", limit: 20 });
  const refreshRef = useRef<HTMLButtonElement>(null);

  const watchlistBySection = useMemo(() => {
    if (!overview?.watchlist_summary.items.length) return new Map<string, typeof overview.watchlist_summary.items>();
    return groupWatchlistItems(overview.watchlist_summary.items, !showDuplicates);
  }, [overview?.watchlist_summary.items, showDuplicates]);

  const visibleWatchlistCount = useMemo(() => {
    let n = 0;
    for (const section of SECTION_ORDER) {
      n += watchlistBySection.get(section)?.length ?? 0;
    }
    return n;
  }, [watchlistBySection]);

  const openAlertsCount = signalsData?.signals?.length ?? 0;
  const ringsDeduped = useMemo(
    () => (overview?.rings_summary ? dedupeRingsForDisplay(overview.rings_summary) : []),
    [overview?.rings_summary]
  );
  const reportsHealthy = overview?.reports_summary?.filter((r) => r.last_run_at).length ?? 0;
  const reportsTotal = overview?.reports_summary?.length ?? 0;

  return (
    <div className="space-y-6">
      {demoMode && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-800 dark:text-amber-200">
          <span className="font-medium">Demo Data Mode</span> – Using fixture data.
        </div>
      )}

      {/* Outcome-first: status, last checked, watching */}
      <div className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">Protection</h1>
        <p className="text-muted-foreground text-sm">
          Is something wrong right now? What it means. What to do next.
        </p>
        {!isLoading && overview && (
          <Card className="rounded-2xl border-border bg-card">
            <CardContent className="py-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3">
                  {openAlertsCount === 0 ? (
                    <>
                      <CheckCircle2 className="h-6 w-6 text-green-600 dark:text-green-500 shrink-0" />
                      <div>
                        <p className="font-medium text-foreground">No urgent threats detected</p>
                        <p className="text-xs text-muted-foreground">
                          Last checked: {overview.last_updated_at ? formatRelative(overview.last_updated_at) : "—"}
                          {visibleWatchlistCount > 0 && ` · Watching: ${visibleWatchlistCount} pattern${visibleWatchlistCount !== 1 ? "s" : ""}`}
                        </p>
                      </div>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="h-6 w-6 text-amber-600 dark:text-amber-500 shrink-0" />
                      <div>
                        <p className="font-medium text-foreground">
                          {openAlertsCount} active situation{openAlertsCount !== 1 ? "s" : ""} need{openAlertsCount === 1 ? "s" : ""} attention
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Last checked: {overview.last_updated_at ? formatRelative(overview.last_updated_at) : "—"}
                        </p>
                      </div>
                    </>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {overview.last_updated_at && (
                    <span className="text-xs text-muted-foreground hidden sm:inline">
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
                  {openAlertsCount > 0 && (
                    <Link href="/alerts">
                      <Button size="sm" className="rounded-xl">
                        View alerts
                      </Button>
                    </Link>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      <LastInvestigationStrip />

      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full rounded-2xl" />
          <Skeleton className="h-64 w-full rounded-2xl" />
        </div>
      ) : overview ? (
        <>
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Left column: what families care about */}
            <div className="lg:col-span-2 space-y-6">
              {openAlertsCount > 0 && (
                <Card className="rounded-2xl border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Active situations</CardTitle>
                    <p className="text-muted-foreground text-sm">
                      Alerts that need your attention. Review and take suggested actions.
                    </p>
                  </CardHeader>
                  <CardContent>
                    <Link href="/alerts">
                      <Button variant="outline" size="sm" className="rounded-xl w-full sm:w-auto">
                        View {openAlertsCount} alert{openAlertsCount !== 1 ? "s" : ""}
                        <ChevronRight className="h-4 w-4 ml-1" />
                      </Button>
                    </Link>
                  </CardContent>
                </Card>
              )}

              <div className="space-y-4">
                <h2 className="text-base font-semibold">What we&apos;re watching</h2>
                <p className="text-muted-foreground text-sm">
                  Topics, contacts, and phrases we monitor to protect you.
                </p>
                {isAdmin && (
                  <div className="flex flex-wrap items-center gap-4">
                    <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showAgentProvenance}
                        onChange={(e) => setShowAgentProvenance(e.target.checked)}
                        className="rounded border-border"
                      />
                      Show agent provenance
                    </label>
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
                )}
                {visibleWatchlistCount === 0 ? (
                  <Card className="rounded-2xl border-border">
                    <CardContent className="py-8 text-center text-sm text-muted-foreground">
                      No watchlist patterns right now. Run an investigation to populate.
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
                          viewAllHref={`/watchlists?category=${section}`}
                          viewAllLabel={`View all ${items.length}`}
                          emptySectionHint={section}
                        />
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Right column: advanced (connected patterns + system checks) */}
            <div className="space-y-6">
              <Card className="rounded-2xl border-border">
                <button
                  type="button"
                  onClick={() => setShowPatterns(!showPatterns)}
                  className="w-full text-left"
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">Connected patterns</CardTitle>
                      <ChevronDown className={cn("h-4 w-4 transition", showPatterns && "rotate-180")} />
                    </div>
                    <p className="text-muted-foreground text-xs">
                      Possible scam networks we&apos;ve detected. Shown when we see repeated contacts + urgency language.
                    </p>
                  </CardHeader>
                </button>
                {showPatterns && (
                  <CardContent className="pt-0">
                    {ringsDeduped.length === 0 ? (
                      <p className="text-sm text-muted-foreground py-4">No connected patterns detected.</p>
                    ) : (
                      <ul className="space-y-2">
                        {ringsDeduped.slice(0, 5).map(({ ring, similarCount }) => (
                          <li key={ring.id}>
                            <RingCard ring={ring} similarCount={similarCount} />
                          </li>
                        ))}
                      </ul>
                    )}
                    {ringsDeduped.length > 0 && (
                      <Link href="/rings" className="block mt-2">
                        <Button variant="ghost" size="sm" className="w-full rounded-xl text-xs">
                          View all patterns
                          <ChevronRight className="h-3 w-3 ml-1" />
                        </Button>
                      </Link>
                    )}
                  </CardContent>
                )}
              </Card>

              <Card className="rounded-2xl border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">System checks</CardTitle>
                  <p className="text-muted-foreground text-xs">
                    Model health, calibration, and safety checks. For technical details, open reports.
                  </p>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {reportsHealthy === reportsTotal && reportsTotal > 0
                      ? "All systems healthy"
                      : reportsHealthy > 0
                        ? `${reportsHealthy} of ${reportsTotal} checks run recently`
                        : "No system checks run yet"}
                  </p>
                  <Link href="/reports" className="inline-block mt-2">
                    <Button variant="outline" size="sm" className="rounded-xl">
                      View reports
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
