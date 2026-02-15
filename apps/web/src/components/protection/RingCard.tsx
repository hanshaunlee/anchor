"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import type { ProtectionRingSummary } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";

const NEW_UPDATED_MS = 30 * 60 * 1000; // 30 min

/** Human-facing risk tier from numeric score (tooltip: based on evidence + similarity + recency). */
export function scoreToRiskTier(score: number): "Low" | "Medium" | "High" {
  if (score >= 0.6) return "High";
  if (score >= 0.3) return "Medium";
  return "Low";
}

export type RingCardProps = {
  ring: ProtectionRingSummary;
  showChangeBadge?: boolean;
  /** When > 0, show "+N similar patterns" instead of duplicate cards */
  similarCount?: number;
  className?: string;
};

export function RingCard({ ring, showChangeBadge = true, similarCount = 0, className }: RingCardProps) {
  const patternLabel = ring.summary_label || patternLabelFromMeta(ring);
  const riskTier = scoreToRiskTier(ring.score);
  const updatedAt = new Date(ring.updated_at).getTime();
  const now = Date.now();
  const isRecent = showChangeBadge && now - updatedAt < NEW_UPDATED_MS;
  const createdAt = new Date(ring.created_at).getTime();
  const isNew = showChangeBadge && now - createdAt < NEW_UPDATED_MS;
  const topEntities = (ring.meta?.top_entities as string[] | Array<{ label?: string }> | undefined) ?? [];
  const includesLine = topEntities.length > 0
    ? topEntities.slice(0, 5).map((e) => typeof e === "string" ? e : (e as { label?: string }).label ?? "").filter(Boolean).join(", ")
    : null;

  return (
    <Link href={`/rings/${ring.id}`}>
      <div
        className={cn(
          "rounded-xl border border-border p-3 hover:bg-muted/30 transition text-left",
          className
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <p className="font-medium text-sm line-clamp-2 flex-1 min-w-0">{patternLabel}</p>
          {(isNew || isRecent) && (
            <Badge variant={isNew ? "default" : "secondary"} className="shrink-0 text-[10px] rounded-md">
              {isNew ? "New" : "Updated"}
            </Badge>
          )}
        </div>
        {ring.summary_text && (
          <p className="text-xs text-muted-foreground mt-1.5 line-clamp-2" title="Why it matters">
            {ring.summary_text}
          </p>
        )}
        {includesLine && (
          <p className="text-xs text-muted-foreground mt-1 line-clamp-1">
            Includes: {includesLine}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-2 mt-2">
          <span
            className={cn(
              "text-[10px] font-medium rounded-md px-1.5 py-0.5",
              riskTier === "High" && "bg-destructive/15 text-destructive",
              riskTier === "Medium" && "bg-amber-500/15 text-amber-700 dark:text-amber-400",
              riskTier === "Low" && "bg-muted text-muted-foreground"
            )}
            >
            Risk: {riskTier}
          </span>
          {similarCount > 0 && (
            <span className="text-[10px] text-muted-foreground">+{similarCount} similar</span>
          )}
        </div>
      </div>
    </Link>
  );
}

/** User-facing pattern label: Top merchant + keyword + org (e.g. "Target + verify identity + IRS"). */
function patternLabelFromMeta(ring: ProtectionRingSummary): string {
  const meta = ring.meta as Record<string, unknown> | undefined;
  const topics = (meta?.topics as string[] | undefined) ?? [];
  const phrases = (meta?.phrases as string[] | undefined) ?? [];
  const topEntities = (meta?.top_entities as string[] | undefined) ?? [];
  const parts: string[] = [];
  if (topics.length) parts.push(topics.slice(0, 2).join(" + "));
  if (phrases.length) parts.push(phrases.slice(0, 2).join(" + "));
  const entityLabels = topEntities
    .slice(0, 2)
    .map((e) => (typeof e === "string" ? e : (e as { label?: string }).label))
    .filter(Boolean);
  if (entityLabels.length) parts.push(entityLabels.join(", "));
  if (parts.length) return parts.join(" · ");
  if (ring.members_count > 0) return `Connected pattern (${ring.members_count} links)`;
  return "Connected pattern";
}
