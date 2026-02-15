"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, List, Send } from "lucide-react";

type ArtifactLinksProps = {
  createdSignalIds?: string[];
  updatedSignalIds?: string[];
  watchlistCount?: number;
  outreachCandidatesCount?: number;
};

export function ArtifactLinks({
  createdSignalIds = [],
  updatedSignalIds = [],
  watchlistCount = 0,
  outreachCandidatesCount = 0,
}: ArtifactLinksProps) {
  const signalIds = [...new Set([...createdSignalIds, ...updatedSignalIds])];
  return (
    <Card className="rounded-2xl shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Artifacts</CardTitle>
        <p className="text-muted-foreground text-sm">Links to created or updated objects</p>
      </CardHeader>
      <CardContent className="space-y-2">
        {signalIds.length > 0 && (
          <div className="flex flex-wrap gap-2 items-center">
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm">Alerts:</span>
            {signalIds.slice(0, 10).map((id) => (
              <Link key={id} href={`/alerts/${id}`}>
                <Button variant="outline" size="sm" className="h-7 text-xs rounded-lg">
                  {id.slice(0, 8)}…
                </Button>
              </Link>
            ))}
            {signalIds.length > 10 && (
              <span className="text-xs text-muted-foreground">+{signalIds.length - 10} more</span>
            )}
          </div>
        )}
        {watchlistCount > 0 && (
          <div className="flex items-center gap-2">
            <List className="h-4 w-4 text-muted-foreground" />
            <Link href="/watchlists">
              <Button variant="outline" size="sm" className="h-7 text-xs rounded-lg">
                Watchlists ({watchlistCount})
              </Button>
            </Link>
          </div>
        )}
        {outreachCandidatesCount > 0 && (
          <div className="flex items-center gap-2">
            <Send className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              {outreachCandidatesCount} outreach candidate(s) — use Actions tab to preview & send
            </span>
          </div>
        )}
        {signalIds.length === 0 && watchlistCount === 0 && outreachCandidatesCount === 0 && (
          <p className="text-sm text-muted-foreground">No artifacts from this run.</p>
        )}
      </CardContent>
    </Card>
  );
}
