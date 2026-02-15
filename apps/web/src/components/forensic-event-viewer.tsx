"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { EventListItem } from "@/lib/api/schemas";
import { cn } from "@/lib/utils";
import {
  Mic,
  Brain,
  Radio,
  Zap,
  MessageSquare,
  Lock,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

const EVENT_TYPE_ICON: Record<string, React.ComponentType<{ className?: string }>> = {
  wake: Zap,
  final_asr: Mic,
  asr: Mic,
  intent: Brain,
  nlu: Brain,
  response: MessageSquare,
  tts: Radio,
  default: MessageSquare,
};

const EVENT_TYPE_COLOR: Record<string, string> = {
  wake: "bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30",
  final_asr: "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/30",
  asr: "bg-blue-500/15 text-blue-700 dark:text-blue-400 border-blue-500/30",
  intent: "bg-violet-500/15 text-violet-700 dark:text-violet-400 border-violet-500/30",
  nlu: "bg-violet-500/15 text-violet-700 dark:text-violet-400 border-violet-500/30",
  response: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30",
  tts: "bg-slate-500/15 text-slate-700 dark:text-slate-400 border-slate-500/30",
  default: "bg-muted text-muted-foreground border-border",
};

export type ForensicEventViewerProps = {
  events: EventListItem[];
  isLoading?: boolean;
  onLoadMore?: () => void;
  hasMore?: boolean;
  className?: string;
};

export function ForensicEventViewer({
  events,
  isLoading,
  onLoadMore,
  hasMore,
  className,
}: ForensicEventViewerProps) {
  return (
    <Card className={cn("rounded-2xl shadow-sm border-border", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Evidence reconstruction</CardTitle>
        <p className="text-muted-foreground text-xs">
          Forensic event stream. Time → sequence; redacted content is masked by consent policy.
        </p>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-16 rounded-xl bg-muted/50 animate-pulse" />
            ))}
          </div>
        ) : events.length === 0 ? (
          <p className="text-muted-foreground py-8 text-center text-sm">No events.</p>
        ) : (
          <div className="relative">
            {/* Time axis label */}
            <div className="flex items-center gap-2 mb-3 text-[10px] uppercase tracking-wider text-muted-foreground font-mono">
              <span>Time →</span>
            </div>
            <ul className="space-y-2">
              {events.map((e, idx) => (
                <ForensicEventNode
                  key={e.id}
                  event={e}
                  prevTs={idx > 0 ? events[idx - 1].ts : null}
                />
              ))}
            </ul>
            {hasMore && onLoadMore && (
              <Button
                variant="outline"
                className="mt-4 w-full rounded-xl"
                onClick={onLoadMore}
              >
                Load more
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ForensicEventNode({
  event,
  prevTs,
}: {
  event: EventListItem;
  prevTs: string | null;
}) {
  const [expanded, setExpanded] = useState(false);
  const typeKey = event.event_type?.toLowerCase().replace(/-/g, "_") ?? "default";
  const Icon = EVENT_TYPE_ICON[typeKey] ?? EVENT_TYPE_ICON.default;
  const colorClass = EVENT_TYPE_COLOR[typeKey] ?? EVENT_TYPE_COLOR.default;
  const ts = new Date(event.ts);
  const deltaMs =
    prevTs != null ? ts.getTime() - new Date(prevTs).getTime() : null;
  const text =
    typeof (event.payload as Record<string, unknown>)?.text === "string"
      ? (event.payload as Record<string, unknown>).text as string
      : null;

  return (
    <li className="rounded-xl border border-border overflow-hidden bg-card">
      <div className="flex gap-3 p-3">
        <div
          className={cn(
            "shrink-0 flex h-9 w-9 items-center justify-center rounded-lg border",
            colorClass
          )}
        >
          <Icon className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
              #{event.seq}
            </span>
            <span className="font-mono text-xs text-foreground">
              {ts.toLocaleTimeString(undefined, {
                hour: "numeric",
                minute: "2-digit",
                second: "2-digit",
              })}
            </span>
            {deltaMs != null && (
              <span className="font-mono text-[10px] text-muted-foreground">
                +{deltaMs}ms
              </span>
            )}
            <span
              className={cn(
                "rounded px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider border",
                colorClass
              )}
            >
              {event.event_type}
            </span>
          </div>
          {event.text_redacted ? (
            <div className="relative mt-2 rounded-lg bg-muted/50 p-3 flex items-center gap-2">
              <div className="absolute inset-0 backdrop-blur-[6px] bg-muted/60 rounded-lg" />
              <Lock className="h-4 w-4 shrink-0 text-muted-foreground relative z-10" />
              <span
                className="text-xs text-muted-foreground italic relative z-10"
                title="Hidden due to consent policy"
              >
                Redacted due to consent policy
              </span>
            </div>
          ) : text ? (
            <p className="text-sm text-foreground mt-1.5 line-clamp-2">{text}</p>
          ) : null}
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 mt-2 text-muted-foreground text-xs hover:text-foreground"
          >
            {expanded ? <ChevronUp className="h-3.5 w-3" /> : <ChevronDown className="h-3.5 w-3" />}
            Payload
          </button>
          {expanded && (
            <pre className="mt-2 text-[11px] font-mono overflow-auto rounded-lg bg-muted/50 p-3 max-h-40 border border-border">
              {JSON.stringify(event.payload, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </li>
  );
}
