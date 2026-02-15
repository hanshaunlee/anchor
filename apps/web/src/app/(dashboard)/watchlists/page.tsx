"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useProtectionWatchlists, useWatchlists } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import {
  WatchlistSectionCard,
  groupWatchlistItems,
} from "@/components/protection/WatchlistSectionCard";
import { getWatchlistDisplay } from "@/lib/watchlist-display";
import { List, ChevronRight } from "lucide-react";

const SECTION_ORDER = ["device_policy", "contact", "phrase", "topic"] as const;

export default function WatchlistsPage() {
  const demoMode = useAppStore((s) => s.demoMode);
  const { data: protectionData, isLoading: protectionLoading } = useProtectionWatchlists({ limit: 200 });
  const { data: legacyData, isLoading: legacyLoading } = useWatchlists();

  const protectionItems = protectionData?.items ?? [];
  const legacyWatchlists = legacyData?.watchlists ?? [];
  const useProtection = protectionItems.length > 0;
  const isLoading = useProtection ? protectionLoading : legacyLoading;

  const watchlistBySection =
    useProtection && protectionItems.length > 0
      ? groupWatchlistItems(protectionItems, true)
      : new Map<string, typeof protectionItems>();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Watchlists</h1>
        <p className="text-muted-foreground text-sm mt-1">
          People and topics the system is watching for in calls and messages to help protect you.
        </p>
        <Link href="/protection" className="text-xs text-primary hover:underline mt-1 inline-flex items-center gap-0.5">
          View on Protection
          <ChevronRight className="h-3 w-3" />
        </Link>
      </div>

      {isLoading ? (
        <Card className="rounded-2xl border-border">
          <CardContent className="py-8">
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16 w-full rounded-xl" />
              ))}
            </div>
          </CardContent>
        </Card>
      ) : useProtection ? (
        <div className="space-y-4">
          <Card className="rounded-2xl border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <List className="h-4 w-4" />
                What we&apos;re watching for
              </CardTitle>
              <p className="text-muted-foreground text-sm">
                Grouped by type. Use the Protection page for a compact overview.
              </p>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              {SECTION_ORDER.map((section) => {
                const items = watchlistBySection.get(section) ?? [];
                const why = items[0]?.explanation ?? undefined;
                return (
                  <WatchlistSectionCard
                    key={section}
                    section={section}
                    items={items}
                    topN={12}
                    why={why}
                    showTypeConfidence
                    viewAllHref="/watchlists"
                  />
                );
              })}
            </CardContent>
          </Card>
        </div>
      ) : legacyWatchlists.length === 0 ? (
        <Card className="rounded-2xl border-border">
          <CardContent className="py-12 text-center text-muted-foreground">
            No watchlists yet. Run an investigation from the Automation page to populate watchlists.
          </CardContent>
        </Card>
      ) : (
        <Card className="rounded-2xl border-border">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <List className="h-4 w-4" />
              What we&apos;re watching for
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {legacyWatchlists.map((w) => {
                const d = getWatchlistDisplay(w);
                return (
                  <li
                    key={w.id}
                    className="rounded-xl border border-border p-4 flex flex-wrap items-start justify-between gap-4"
                  >
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-medium text-sm">{d.title}</p>
                        {d.modelAvailable && (
                          <Badge variant="secondary" className="text-xs rounded-md">
                            AI pattern
                          </Badge>
                        )}
                      </div>
                      {d.reason ? (
                        <p className="text-muted-foreground text-sm">{d.reason}</p>
                      ) : (
                        <p className="text-muted-foreground text-sm">{d.detail}</p>
                      )}
                      {d.reason && d.detail && (w.watch_type === "keyword" || w.watch_type === "merchant" || w.watch_type === "embedding_centroid") && (
                        <p className="text-muted-foreground text-xs">{d.detail}</p>
                      )}
                      <p className="text-muted-foreground text-xs">
                        Priority {d.priority}
                        {d.expires && ` · ${d.expires}`}
                      </p>
                    </div>
                  </li>
                );
              })}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
