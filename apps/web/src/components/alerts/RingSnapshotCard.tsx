"use client";

import React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useProtectionRing } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Network, ExternalLink } from "lucide-react";

export type RingSnapshotCardProps = {
  ringId: string;
};

export function RingSnapshotCard({ ringId }: RingSnapshotCardProps) {
  const { data: ring, isLoading } = useProtectionRing(ringId);

  if (isLoading) {
    return (
      <Card className="rounded-2xl border-border">
        <CardContent className="py-4">
          <Skeleton className="h-16 w-full rounded-lg" />
        </CardContent>
      </Card>
    );
  }

  if (!ring) {
    return (
      <Card className="rounded-2xl border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Network className="h-4 w-4 text-muted-foreground" />
            Ring snapshot
          </CardTitle>
          <p className="text-sm text-muted-foreground">Ring details unavailable (e.g. demo mode). You can still open the ring in the graph or view the ring page.</p>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Link href={`/graph?highlightRing=${encodeURIComponent(ringId)}`}>
            <Button variant="outline" size="sm" className="rounded-xl">
              <Network className="h-4 w-4 mr-2" />
              Open in Graph
            </Button>
          </Link>
          <Link href={`/rings/${ringId}`}>
            <Button variant="ghost" size="sm" className="rounded-xl">
              View ring
              <ExternalLink className="h-3 w-3 ml-1" />
            </Button>
          </Link>
        </CardContent>
      </Card>
    );
  }

  const members = ring.members ?? [];
  const title = ring.summary_label ?? (members.length ? `Cluster of ${members.length} entities` : "Ring");

  return (
    <Card className="rounded-2xl border-border">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Network className="h-4 w-4 text-muted-foreground" />
          Ring snapshot
        </CardTitle>
        <p className="font-medium text-sm text-foreground">{title}</p>
        {ring.summary_text && (
          <p className="text-xs text-muted-foreground line-clamp-2">{ring.summary_text}</p>
        )}
        <p className="text-xs text-muted-foreground">
          {members.length} members · score {(ring as { score?: number }).score?.toFixed(2) ?? "—"}
        </p>
      </CardHeader>
      <CardContent className="flex flex-wrap gap-2">
        <Link href={`/graph?highlightRing=${encodeURIComponent(ringId)}`}>
          <Button variant="outline" size="sm" className="rounded-xl">
            <Network className="h-4 w-4 mr-2" />
            Open in Graph
          </Button>
        </Link>
        <Link href={`/rings/${ringId}`}>
          <Button variant="ghost" size="sm" className="rounded-xl">
            View ring
            <ExternalLink className="h-3 w-3 ml-1" />
          </Button>
        </Link>
      </CardContent>
    </Card>
  );
}
