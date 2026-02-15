"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { ProtectionWatchlistItem } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";
import { ChevronRight } from "lucide-react";

const SECTION_ORDER = ["device_policy", "contact", "phrase", "topic"] as const;
const SECTION_LABELS: Record<string, string> = {
  device_policy: "Device protections",
  contact: "Contacts to watch",
  phrase: "Phrases",
  topic: "Topics",
  bank: "Bank",
  other: "Other",
};

const CONFIDENCE_LABEL = (score: number | null): string => {
  if (score == null) return "";
  if (score >= 0.7) return "High";
  if (score >= 0.4) return "Med";
  return "Low";
};

const EMPTY_HINTS: Record<string, string> = {
  device_policy:
    "No device protections set. These appear when you enable high-risk mode or similar from Protection.",
  phrase:
    "No risky phrases added yet. Agents currently add contacts and topics; phrase rules can be added later.",
  topic: "No topic patterns yet. Run an investigation to detect risky topics.",
  contact: "No contacts on the watchlist yet. Run an investigation to flag numbers or emails.",
};

export type WatchlistSectionCardProps = {
  /** Section key: device_policy | contact | phrase | topic */
  section: string;
  items: ProtectionWatchlistItem[];
  /** Max chips to show (default 6) */
  topN?: number;
  /** Single "why" line for the section */
  why?: string;
  lastUpdated?: string | null;
  /** Show type + confidence on each chip */
  showTypeConfidence?: boolean;
  /** When true (admin/judge): show source agents for this section */
  showAgentProvenance?: boolean;
  viewAllHref?: string;
  viewAllLabel?: string;
  /** When set, show this section's empty-state hint when items.length === 0 */
  emptySectionHint?: string;
  className?: string;
};

export function WatchlistSectionCard({
  section,
  items,
  topN = 6,
  why,
  lastUpdated,
  showTypeConfidence = true,
  showAgentProvenance = false,
  viewAllHref = "/watchlists",
  viewAllLabel,
  emptySectionHint,
  className,
}: WatchlistSectionCardProps) {
  const title = SECTION_LABELS[section] ?? section;
  const displayItems = items.slice(0, topN);
  const total = items.length;
  const sources = showAgentProvenance
    ? [...new Set(items.map((i) => i.source_agent).filter(Boolean))] as string[]
    : [];
  const emptyHint = emptySectionHint ? EMPTY_HINTS[emptySectionHint] : undefined;

  return (
    <Card className={cn("rounded-2xl border-border", className)}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">{title}</CardTitle>
          <span className="text-xs text-muted-foreground tabular-nums">
            {total} {lastUpdated ? `· ${formatRelative(lastUpdated)}` : ""}
          </span>
        </div>
        {why && <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">{why}</p>}
        {showAgentProvenance && sources.length > 0 && (
          <p className="text-[10px] text-muted-foreground mt-0.5">Source: {sources.join(", ")}</p>
        )}
      </CardHeader>
      <CardContent className="space-y-2">
        {displayItems.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            {emptyHint ?? "None"}
          </p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {displayItems.map((item) => (
              <Badge
                key={item.id}
                variant="secondary"
                className="rounded-lg text-xs font-normal py-1 px-2"
                title={item.explanation ?? undefined}
              >
                <span>{item.display_value || item.display_label || item.type}</span>
                {showTypeConfidence && (
                  <span className="ml-1 opacity-70">
                    {item.display_label ?? item.type}
                    {item.score != null && ` · ${CONFIDENCE_LABEL(item.score)}`}
                  </span>
                )}
              </Badge>
            ))}
          </div>
        )}
        {total > topN && (
          <Link href={viewAllHref}>
            <Button variant="ghost" size="sm" className="h-7 text-xs rounded-lg px-2 -ml-2">
              View all ({total})
              <ChevronRight className="h-3 w-3 ml-0.5" />
            </Button>
          </Link>
        )}
      </CardContent>
    </Card>
  );
}

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

/** Group items by category and optionally dedupe by (type + display_value). */
export function groupWatchlistItems(
  items: ProtectionWatchlistItem[],
  dedupe: boolean
): Map<string, ProtectionWatchlistItem[]> {
  const seen = new Set<string>();
  const filtered = dedupe
    ? items.filter((item) => {
        const key = `${item.type}|${(item.display_value ?? "").trim().toLowerCase()}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
    : items;
  const map = new Map<string, ProtectionWatchlistItem[]>();
  for (const item of filtered) {
    const cat = item.category || "other";
    if (!map.has(cat)) map.set(cat, []);
    map.get(cat)!.push(item);
  }
  for (const cat of SECTION_ORDER) {
    if (!map.has(cat)) map.set(cat, []);
  }
  return map;
}
