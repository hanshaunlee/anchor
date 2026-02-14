/**
 * Human-friendly display for watchlist entries. Backend stores watch_type + pattern (jsonb);
 * we never show raw JSON to users—we derive labels and summaries from the schema.
 */

import type { WatchlistItem } from "@/lib/api/schemas";

const WATCH_TYPE_LABELS: Record<string, string> = {
  entity_pattern: "Entity / contact",
  keyword: "Topic / keyword",
  embedding_centroid: "Behavior pattern",
  merchant: "Merchant",
  entity_hash: "Contact (hashed)",
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
      detail: list ? `Keywords: ${list}` : "Topic-based pattern",
    };
  }

  if (w.watch_type === "entity_pattern" || w.watch_type === "entity_hash") {
    const entityType = p.entity_type as string | undefined;
    const score = p.score as number | undefined;
    const nodeIndex = p.node_index as number | undefined;
    const parts: string[] = [];
    if (entityType) parts.push(`Type: ${entityType}`);
    if (score != null) parts.push(`Relevance: ${Math.round(score * 100)}%`);
    if (nodeIndex != null && parts.length === 0) parts.push(`Entity #${nodeIndex + 1}`);
    return {
      title: typeLabel,
      detail: parts.length ? parts.join(" · ") : "Entity match pattern",
    };
  }

  if (w.watch_type === "embedding_centroid") {
    const pat = w.pattern ?? {};
    const threshold = pat.threshold as number | undefined;
    const source = pat.source as { window?: string; risk_signal_ids?: string[] } | undefined;
    const prov = pat.provenance as { window_days?: number; risk_signal_ids?: string[] } | undefined;
    const window = source?.window ?? (prov?.window_days != null ? `${prov.window_days}d` : null);
    const count = (source?.risk_signal_ids?.length ?? prov?.risk_signal_ids?.length ?? 0) || null;
    const parts: string[] = [];
    if (threshold != null) parts.push(`threshold ${(threshold * 100).toFixed(0)}%`);
    if (window) parts.push(`window ${window}`);
    if (count != null && count > 0) parts.push(`${count} signals`);
    return {
      title: w.model_available === true ? "GNN centroid · Behavior pattern" : typeLabel,
      detail: parts.length > 0 ? parts.join(" · ") : "Similarity-based behavior pattern (requires model embeddings)",
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
  const keys = Object.keys(p).filter((k) => typeof p[k] !== "object" || Array.isArray(p[k]));
  const short =
    keys.length > 0
      ? keys
          .slice(0, 3)
          .map((k) => `${k}: ${Array.isArray(p[k]) ? (p[k] as unknown[]).slice(0, 3).join(", ") : String(p[k])}`)
          .join(" · ")
      : "Pattern";
  return { title: typeLabel, detail: short };
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
