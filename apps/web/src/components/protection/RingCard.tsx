"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import type { ProtectionRingSummary } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";

const NEW_UPDATED_MS = 30 * 60 * 1000; // 30 min

export type RingCardProps = {
  ring: ProtectionRingSummary;
  /** Optional: show "New" or "Updated" badge */
  showChangeBadge?: boolean;
  className?: string;
};

export function RingCard({ ring, showChangeBadge = true, className }: RingCardProps) {
  const title = ring.summary_label || heuristicTitle(ring);
  const updatedAt = new Date(ring.updated_at).getTime();
  const now = Date.now();
  const isRecent = showChangeBadge && now - updatedAt < NEW_UPDATED_MS;
  const createdAt = new Date(ring.created_at).getTime();
  const isNew = showChangeBadge && now - createdAt < NEW_UPDATED_MS;
  const topEntities = (ring.meta?.top_entities as string[] | undefined) ?? [];

  return (
    <Link href={`/rings/${ring.id}`}>
      <div
        className={cn(
          "rounded-xl border border-border p-2.5 hover:bg-muted/30 transition",
          className
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <p className="font-medium text-sm line-clamp-2 flex-1 min-w-0">{title}</p>
          {(isNew || isRecent) && (
            <Badge variant={isNew ? "default" : "secondary"} className="shrink-0 text-[10px] rounded-md">
              {isNew ? "New" : "Updated"}
            </Badge>
          )}
        </div>
        {ring.summary_text && (
          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{ring.summary_text}</p>
        )}
        {topEntities.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {topEntities.slice(0, 4).map((e, i) => (
              <span
                key={i}
                className="inline-flex items-center rounded-md bg-muted px-1.5 py-0.5 text-[10px]"
              >
                {typeof e === "string" ? e : (e as { label?: string }).label ?? String(e)}
              </span>
            ))}
          </div>
        )}
        <p className="text-[10px] text-muted-foreground mt-1.5">
          {ring.members_count} members · score {ring.score.toFixed(2)}
        </p>
      </div>
    </Link>
  );
}

function heuristicTitle(ring: ProtectionRingSummary): string {
  const meta = ring.meta as Record<string, unknown> | undefined;
  const topics = meta?.topics as string[] | undefined;
  const phrases = meta?.phrases as string[] | undefined;
  if (topics?.length) {
    const joined = topics.slice(0, 2).join(" + ");
    return `Cluster around ${joined}`;
  }
  if (phrases?.length) {
    const joined = phrases.slice(0, 2).join(" + ");
    return `Cluster around ${joined}`;
  }
  if (ring.members_count > 0) {
    return `Cluster of ${ring.members_count} entities`;
  }
  return "Ring cluster";
}
