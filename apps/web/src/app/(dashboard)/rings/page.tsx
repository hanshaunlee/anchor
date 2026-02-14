"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useRings } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Network } from "lucide-react";

export default function RingsPage() {
  const { data, isLoading } = useRings();
  const rings = data?.rings ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Rings</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Ring Discovery agent: clustered entity groups and connector analysis. Open a ring to see members.
        </p>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Network className="h-4 w-4" />
            Discovered rings
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16 w-full rounded-xl" />
              ))}
            </div>
          ) : rings.length === 0 ? (
            <p className="text-muted-foreground py-12 text-center">No rings yet. Run the Ring Discovery agent from the Agents page.</p>
          ) : (
            <ul className="space-y-3">
              {rings.map((ring) => {
                const meta = (ring.meta ?? {}) as { member_count?: number; index?: number; top_connectors?: unknown[] };
                return (
                  <li
                    key={ring.id}
                    className="rounded-2xl border border-border p-4 flex flex-wrap items-center justify-between gap-4"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="font-medium text-sm">Ring · score {ring.score.toFixed(2)}</p>
                      <p className="text-muted-foreground text-xs">
                        {meta.member_count != null ? `${meta.member_count} members` : ""}
                        {ring.created_at ? ` · ${new Date(ring.created_at).toLocaleString()}` : ""}
                      </p>
                    </div>
                    <Link href={`/rings/${ring.id}`}>
                      <Button variant="outline" size="sm">
                        View ring
                      </Button>
                    </Link>
                  </li>
                );
              })}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
