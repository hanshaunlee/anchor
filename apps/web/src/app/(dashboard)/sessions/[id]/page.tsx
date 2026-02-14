"use client";

import { use, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useSessions, useSessionEvents } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowLeft, ChevronDown, ChevronUp } from "lucide-react";

export default function SessionDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data: sessionsData } = useSessions({ limit: 100 });
  const session = sessionsData?.sessions?.find((s) => s.id === id);
  const { data: eventsData, isLoading } = useSessionEvents(id, { limit: 50 });
  const [offset, setOffset] = useState(0);
  const { data: nextPage } = useSessionEvents(id, { limit: 50, offset: offset + 50 });
  const events = eventsData?.events ?? [];
  const total = eventsData?.total ?? 0;
  const hasMore = nextPage && nextPage.events.length > 0;

  if (!session) {
    return (
      <div className="space-y-6">
        <Link href="/sessions" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:underline">
          <ArrowLeft className="h-4 w-4" /> Back to sessions
        </Link>
        <p className="text-muted-foreground">Session not found.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-4">
        <Link href="/sessions" className="rounded-xl p-2 hover:bg-accent">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Session</h1>
          <p className="text-muted-foreground text-sm">
            {new Date(session.started_at).toLocaleString()} – {session.mode}
          </p>
        </div>
      </div>

      {session.summary_text && (
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{session.summary_text}</p>
          </CardContent>
        </Card>
      )}

      {session.consent_state && Object.keys(session.consent_state).length > 0 && (
        <Card className="rounded-2xl shadow-sm border-amber-200 bg-amber-50/50 dark:border-amber-900 dark:bg-amber-950/20">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Consent state</CardTitle>
            <p className="text-muted-foreground text-sm">What is redacted and why</p>
          </CardHeader>
          <CardContent>
            <pre className="text-xs overflow-auto rounded-xl bg-background p-3">
              {JSON.stringify(session.consent_state, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Events ({total})</CardTitle>
          <p className="text-muted-foreground text-sm">Paginated; redacted by consent</p>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-16 w-full rounded-xl" />
              ))}
            </div>
          ) : events.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">No events.</p>
          ) : (
            <ul className="space-y-3">
              {events.map((e) => (
                <li key={e.id} className="rounded-xl border border-border overflow-hidden">
                  <div className="p-4 flex justify-between items-start gap-2">
                    <div className="min-w-0">
                      <p className="text-sm font-medium">{e.event_type}</p>
                      <p className="text-muted-foreground text-xs">
                        {new Date(e.ts).toLocaleString()} · seq {e.seq}
                      </p>
                      {e.text_redacted ? (
                        <p className="text-muted-foreground italic text-sm mt-1">
                          Redacted due to consent
                        </p>
                      ) : (
                        <p className="text-sm mt-1 truncate">
                          {typeof (e.payload as Record<string, unknown>)?.text === "string"
                            ? (e.payload as Record<string, unknown>).text as string
                            : null}
                        </p>
                      )}
                    </div>
                    <EventPayloadCollapse payload={e.payload} />
                  </div>
                </li>
              ))}
            </ul>
          )}
          {hasMore && (
            <Button
              variant="outline"
              className="mt-4 w-full rounded-xl"
              onClick={() => setOffset(offset + 50)}
            >
              Load more
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function EventPayloadCollapse({ payload }: { payload: Record<string, unknown> }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="shrink-0">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-muted-foreground text-xs hover:text-foreground"
      >
        {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        Payload
      </button>
      {open && (
        <pre className="mt-2 text-xs overflow-auto rounded-lg bg-muted p-2 max-h-40">
          {JSON.stringify(payload, null, 2)}
        </pre>
      )}
    </div>
  );
}
