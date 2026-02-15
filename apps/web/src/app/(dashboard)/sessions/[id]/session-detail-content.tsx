"use client";

import { useState } from "react";
import Link from "next/link";
import { useSessions, useSessionEvents, useHouseholdConsent } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Skeleton } from "@/components/ui/skeleton";
import { SessionIntelligenceSummary } from "@/components/session-intelligence-summary";
import { ForensicEventViewer } from "@/components/forensic-event-viewer";
import { SessionAgentInterpretation } from "@/components/session-agent-interpretation";
import { ArrowLeft } from "lucide-react";

export function SessionDetailContent({ id }: { id: string }) {
  const demoMode = useAppStore((s) => s.demoMode);
  const { data: sessionsData } = useSessions({ limit: 200 });
  const session = sessionsData?.sessions?.find((s) => s.id === id);
  const { data: eventsData, isLoading } = useSessionEvents(id, { limit: 50 });
  const { data: consent } = useHouseholdConsent();
  const [offset, setOffset] = useState(0);
  const { data: nextPage } = useSessionEvents(id, { limit: 50, offset: offset + 50 });
  const events = eventsData?.events ?? [];
  const total = eventsData?.total ?? 0;
  const hasMore = nextPage && nextPage.events.length > 0;

  if (!session) {
    return (
      <div className="space-y-6">
        <Link href="/sessions" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:underline">
          <ArrowLeft className="h-4 w-4" /> Back to data capture timeline
        </Link>
        <p className="text-muted-foreground">Session not found.</p>
      </div>
    );
  }

  const handleLoadMore = () => setOffset(offset + 50);

  return (
    <div className="space-y-8">
      {demoMode && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-800 dark:text-amber-200">
          <span className="font-medium">Demo Data Mode</span> – Using fixture data.
        </div>
      )}

      <div className="flex items-center gap-4">
        <Link href="/sessions" className="rounded-xl p-2 hover:bg-accent">
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Forensic trace viewer</h1>
          <p className="text-muted-foreground text-sm font-mono">
            {new Date(session.started_at).toLocaleString()} – {session.mode}
          </p>
        </div>
      </div>

      {/* A. Executive Summary */}
      <section>
        <SessionIntelligenceSummary
          session={session}
          outboundEligible={consent?.allow_outbound_contact ?? false}
        />
      </section>

      {/* B. Evidence Reconstruction */}
      <section>
        <ForensicEventViewer
          events={events}
          isLoading={isLoading}
          onLoadMore={handleLoadMore}
          hasMore={hasMore}
        />
      </section>

      {/* C. Agent Interpretation */}
      <section>
        <SessionAgentInterpretation />
      </section>
    </div>
  );
}
