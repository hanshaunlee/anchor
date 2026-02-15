"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useExplainMutation } from "@/hooks/use-api";
import { MessageSquare, Loader2, AlertCircle } from "lucide-react";

const API_ITEMS_LIMIT = 20;

export type ExplainableIdsProps = {
  context: "pattern_members" | "alert_ids" | "top_connectors" | "entity_list";
  items: Array<{ id: string; label?: string | null }>;
  /** Section title when explanations are shown */
  title?: string;
  className?: string;
};

/**
 * Renders a list of opaque IDs with an "Explain in plain language" button.
 * On success, shows short explanations from the Explain API (Claude when configured).
 */
export function ExplainableIds({ context, items, title = "What these are", className }: ExplainableIdsProps) {
  const [explained, setExplained] = useState<Array<{ original: string; explanation: string }> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const explainMut = useExplainMutation();

  const handleExplain = () => {
    if (items.length === 0) return;
    setError(null);
    const toSend = items.slice(0, API_ITEMS_LIMIT);
    explainMut.mutate(
      {
        context,
        items: toSend.map((it) => ({ id: it.id, hint: it.label ?? undefined })),
      },
      {
        onSuccess: (data) => {
          const list = Array.isArray(data?.explanations) ? data.explanations : [];
          setExplained(list);
        },
        onError: (err) => {
          const message = err instanceof Error ? err.message : "Couldn't load explanations.";
          setError(message.includes("401") || message.includes("Unauthorized") ? "Sign in to get explanations." : message);
        },
      }
    );
  };

  if (items.length === 0) return null;

  const itemsForDisplay = items.slice(0, API_ITEMS_LIMIT);

  return (
    <div className={className}>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="rounded-lg text-xs"
          onClick={handleExplain}
          disabled={explainMut.isPending}
        >
          {explainMut.isPending ? (
            <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
          ) : (
            <MessageSquare className="h-3.5 w-3.5 mr-1.5" />
          )}
          Explain in plain language
        </Button>
      </div>
      {error && (
        <p className="mt-2 text-xs text-destructive flex items-center gap-1.5 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          {error}
        </p>
      )}
      {explained && explained.length > 0 && (
        <ul className="mt-2 space-y-1.5 rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm">
          <p className="text-xs font-medium text-muted-foreground mb-1">{title}</p>
          {explained.map((e, i) => (
            <li key={`${e.original}-${i}`} className="flex gap-2">
              <span className="text-muted-foreground shrink-0 font-mono text-xs">
                {itemsForDisplay[i]?.label ?? e.original.slice(0, 8)}…
              </span>
              <span className="text-foreground">{e.explanation}</span>
            </li>
          ))}
        </ul>
      )}
      {explained && explained.length === 0 && !error && (
        <p className="mt-2 text-xs text-muted-foreground">No explanations returned. Try again later.</p>
      )}
    </div>
  );
}
