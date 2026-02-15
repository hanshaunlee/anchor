"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { ArrowLeft, Network, ExternalLink, ChevronDown, ChevronRight } from "lucide-react";

type RingMember = { entity_id: string | null; role: string | null; first_seen_at: string | null; last_seen_at: string | null; display_label?: string };
type RingDetail = {
  id: string;
  household_id: string;
  created_at: string;
  updated_at: string;
  score: number;
  meta: Record<string, unknown>;
  members: RingMember[];
  summary_label?: string;
  summary_text?: string;
};

const WHAT_TO_DO_DEFAULT = [
  "Do not buy gift cards or send money based on phone or email requests.",
  "Call the official number from your bank's or agency's website—never use a number from a call or message.",
  "Block and report suspicious numbers. You can say no and hang up.",
];

function isTechnicalId(label: string): boolean {
  const raw = (label || "").toLowerCase();
  return raw.startsWith("acc_") || raw.startsWith("entity (") || /^[0-9a-f-]{8,}$/i.test(raw.replace(/…$/, "").trim());
}

function groupMembersByType(members: RingMember[], showTechnical: boolean): { People: string[]; Numbers: string[]; Keywords: string[]; Accounts: string[] } {
  const People: string[] = [];
  const Numbers: string[] = [];
  const Keywords: string[] = [];
  const Accounts: string[] = [];
  for (const m of members) {
    const label = m.display_label ?? (m.entity_id ? `${String(m.entity_id).slice(0, 8)}…` : "—");
    if (!showTechnical && isTechnicalId(label)) {
      Accounts.push(label);
      continue;
    }
    if (/^\+?[\d\s\-()]{10,}$/.test(label.replace(/\s/g, ""))) {
      Numbers.push(label);
    } else if (label.includes("…") && label.length <= 12) {
      Accounts.push(label);
    } else if (label.length <= 30 && !label.startsWith("entity")) {
      People.push(label);
    } else {
      Keywords.push(label);
    }
  }
  return { People, Numbers, Keywords, Accounts };
}

export default function RingDetailPage({ params }: { params: { id: string } }) {
  const [ring, setRing] = useState<RingDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showTechnical, setShowTechnical] = useState(false);
  const [explainOpen, setExplainOpen] = useState(false);

  const load = useCallback(async () => {
    if (!params.id) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.getProtectionRing(params.id) as RingDetail;
      setRing(data);
    } catch {
      try {
        const legacy = await api.getRing(params.id);
        setRing({
          ...legacy,
          members: (legacy.members ?? []).map((m: RingMember) => ({ ...m, display_label: m.entity_id ? `${String(m.entity_id).slice(0, 8)}…` : "—" })),
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load");
      }
    } finally {
      setLoading(false);
    }
  }, [params.id]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  if (error || !ring) {
    return (
      <div className="space-y-4">
        <Link href="/rings">
          <Button variant="ghost" size="sm" className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to patterns
          </Button>
        </Link>
        <Card className="rounded-2xl border-border">
          <CardContent className="pt-6">
            <p className="text-destructive">{error ?? "Not found."}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const meta = (ring.meta ?? {}) as { member_count?: number; top_connectors?: Array<{ entity_id?: string; betweenness?: number; bridge_risk?: number }>; evidence_edge_count?: number };
  const headline = ring.summary_label || ring.summary_text || "This looks like a connected pattern that may be worth reviewing.";
  const whyBullets = ring.summary_text
    ? [ring.summary_text]
    : meta.member_count != null
      ? [`Repeated contacts or links (${meta.member_count} entities in this pattern).`, "Often seen with urgency language or gift card mentions.", "Review the entities and sessions below."];
  const grouped = groupMembersByType(ring.members ?? [], showTechnical);
  const hasAccounts = grouped.Accounts.length > 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4">
        <Link href="/rings">
          <Button variant="ghost" size="sm" className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to patterns
          </Button>
        </Link>
        <Link href="/protection" className="text-xs text-primary hover:underline inline-flex items-center gap-0.5">
          Protection
          <ExternalLink className="h-3 w-3" />
        </Link>
      </div>

      {/* Plain-English headline */}
      <Card className="rounded-2xl border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg font-semibold leading-snug">
            {headline}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Why we think that */}
          <div>
            <h3 className="text-sm font-medium text-foreground mb-2">Why we think that</h3>
            <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
              {whyBullets.slice(0, 3).map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>

          {/* What to do */}
          <div>
            <h3 className="text-sm font-medium text-foreground mb-2">What to do</h3>
            <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
              {WHAT_TO_DO_DEFAULT.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          </div>

          {/* Entities grouped */}
          <div className="space-y-4">
            {grouped.People.length > 0 && (
              <div>
                <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">People</h3>
                <div className="flex flex-wrap gap-2">
                  {grouped.People.map((label, i) => (
                    <span key={i} className="rounded-lg bg-muted px-2 py-1 text-sm">
                      {label}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {grouped.Numbers.length > 0 && (
              <div>
                <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Numbers</h3>
                <div className="flex flex-wrap gap-2">
                  {grouped.Numbers.map((label, i) => (
                    <span key={i} className="rounded-lg bg-muted px-2 py-1 text-sm font-mono">
                      {label}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {grouped.Keywords.length > 0 && (
              <div>
                <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Keywords / topics</h3>
                <div className="flex flex-wrap gap-2">
                  {grouped.Keywords.map((label, i) => (
                    <span key={i} className="rounded-lg bg-muted px-2 py-1 text-sm">
                      {label}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {(hasAccounts || showTechnical) && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Accounts / technical</h3>
                  {!showTechnical && hasAccounts && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 text-xs rounded-lg"
                      onClick={() => setShowTechnical(true)}
                    >
                      Show technical identifiers
                    </Button>
                  )}
                </div>
                {showTechnical && grouped.Accounts.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {grouped.Accounts.map((label, i) => (
                      <span key={i} className="rounded-lg bg-muted/70 px-2 py-1 text-xs font-mono">
                        {label}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex flex-wrap gap-2 pt-2">
            <Link href={`/graph?highlightRing=${ring.id}`}>
              <Button variant="outline" size="sm" className="rounded-xl">
                <Network className="h-4 w-4 mr-2" />
                Open in Graph
              </Button>
            </Link>
            <Link href="/alerts">
              <Button variant="outline" size="sm" className="rounded-xl">
                View related alerts
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>

      {/* Technical details (expandable) */}
      <Card className="rounded-2xl border-border">
        <button
          type="button"
          onClick={() => setExplainOpen(!explainOpen)}
          className="w-full text-left rounded-2xl hover:bg-muted/30 transition"
        >
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Technical details
              </CardTitle>
              {explainOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            </div>
          </CardHeader>
        </button>
        {explainOpen && (
          <CardContent className="pt-0 space-y-4">
            <p className="text-xs text-muted-foreground">
              Score {ring.score.toFixed(2)}
              {meta.member_count != null ? ` · ${meta.member_count} members` : ""}
              {meta.evidence_edge_count != null ? ` · ${meta.evidence_edge_count} evidence edges` : ""}
            </p>
            {meta.top_connectors && meta.top_connectors.length > 0 && (
              <div>
                <p className="text-xs font-medium text-muted-foreground mb-1">Top connectors</p>
                <ul className="text-xs text-muted-foreground space-y-0.5">
                  {meta.top_connectors.map((c, i) => (
                    <li key={i}>
                      {c.entity_id ? `${String(c.entity_id).slice(0, 8)}…` : "—"}
                      {c.betweenness != null ? ` · betweenness ${c.betweenness.toFixed(2)}` : ""}
                      {c.bridge_risk != null ? ` · bridge risk ${c.bridge_risk.toFixed(2)}` : ""}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        )}
      </Card>
    </div>
  );
}
