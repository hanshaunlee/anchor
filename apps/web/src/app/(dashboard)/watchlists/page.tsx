"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useWatchlists } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { List } from "lucide-react";

export default function WatchlistsPage() {
  const { data, isLoading } = useWatchlists();
  const watchlists = data?.watchlists ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Watchlists</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Patterns pushed to device. Delta view and export read-only unless API supports edit.
        </p>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <List className="h-4 w-4" />
            Watchlist entries
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16 w-full rounded-xl" />
              ))}
            </div>
          ) : watchlists.length === 0 ? (
            <p className="text-muted-foreground py-12 text-center">No watchlists.</p>
          ) : (
            <ul className="space-y-3">
              {watchlists.map((w) => (
                <li
                  key={w.id}
                  className="rounded-2xl border border-border p-4 flex flex-wrap items-center justify-between gap-2"
                >
                  <div>
                    <p className="font-medium text-sm">{w.watch_type}</p>
                    <p className="text-muted-foreground text-xs">{w.reason ?? "—"}</p>
                    <p className="text-muted-foreground text-xs mt-1">
                      Priority {w.priority}
                      {w.expires_at && ` · Expires ${new Date(w.expires_at).toLocaleDateString()}`}
                    </p>
                  </div>
                  <pre className="text-xs rounded-lg bg-muted p-2 max-w-xs overflow-auto">
                    {JSON.stringify(w.pattern)}
                  </pre>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
