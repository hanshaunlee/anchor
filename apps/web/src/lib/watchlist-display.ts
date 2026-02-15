/**
 * Human-friendly display for watchlist entries. Backend stores watch_type + pattern (jsonb);
 * we never show raw JSON to users—we derive labels and summaries from the schema.
 */

import type { WatchlistItem } from "@/lib/api/schemas";

const WATCH_TYPE_LABELS: Record<string, string> = {
  entity_pattern: "Contact to watch",
  keyword: "Risky topics",
  embedding_centroid: "Concerning behavior pattern",
  merchant: "Risky payment phrases",
  entity_hash: "Contact to watch",
};

function patternSummary(w: WatchlistItem): { title: string; detail: string } {
  const p = w.pattern ?? {};
  const typeLabel = WATCH_TYPE_LABELS[w.watch_type] ?? w.watch_type;

  if (w.watch_type === "keyword") {
    const kw = p.keywords;
    const list = Array.isArray(kw)
      ? kw.join(", ")
      : typeof kw === "string"
        ? kw
        : "";
    return {
      title: typeLabel,
      detail: list ? list : "Topic-based pattern",
    };
  }

  if (w.watch_type === "entity_pattern" || w.watch_type === "entity_hash") {
    const entityType = p.entity_type as string | undefined;
    const score = p.score as number | undefined;
    if (entityType && score != null) {
      return {
        title: typeLabel,
        detail: `Flagged contact (high relevance)`,
      };
    }
    if (score != null) {
      return {
        title: typeLabel,
        detail: "Flagged contact (high relevance)",
      };
    }
    return {
      title: typeLabel,
      detail: "Contact or caller flagged by the system",
    };
  }

  if (w.watch_type === "embedding_centroid") {
    const pat = w.pattern ?? {};
    const source = pat.source as { risk_signal_ids?: string[] } | undefined;
    const prov = pat.provenance as { risk_signal_ids?: string[] } | undefined;
    const count = source?.risk_signal_ids?.length ?? prov?.risk_signal_ids?.length ?? 0;
    const detail =
      count > 0
        ? `Based on ${count} recent similar alert${count !== 1 ? "s" : ""}`
        : "Based on similar conversation patterns";
    return {
      title: w.model_available === true ? "Concerning behavior pattern" : typeLabel,
      detail,
    };
  }

  if (w.watch_type === "merchant") {
    const kw = p.keywords;
    const list = Array.isArray(kw)
      ? kw.join(", ")
      : typeof kw === "string"
        ? kw
        : "";
    return {
      title: typeLabel,
      detail: list ? list : "Merchant pattern",
    };
  }

  // Fallback: show a short, safe summary (no raw JSON)
  return { title: typeLabel, detail: "Watchlist entry from the system" };
}

export function getWatchlistDisplay(w: WatchlistItem) {
  const { title, detail } = patternSummary(w);
  const reason = w.reason?.trim() || null;
  const expires = w.expires_at
    ? `Expires ${new Date(w.expires_at).toLocaleDateString()}`
    : null;
  const modelAvailable = w.model_available === true;
  return { title, detail, reason, priority: w.priority, expires, modelAvailable };
}
