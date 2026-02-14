"use client";

import { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useSessions } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Calendar } from "lucide-react";

export default function SessionsPage() {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const { data, isLoading } = useSessions({
    from: from || undefined,
    to: to || undefined,
    limit: 50,
  });

  const sessions = data?.sessions ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Sessions</h1>
        <p className="text-muted-foreground text-sm mt-1">
          List sessions with date range. Consent state indicates what is redacted.
        </p>
      </div>

      <div className="flex flex-wrap gap-4">
        <div className="space-y-2">
          <Label className="text-xs">From</Label>
          <Input
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            className="rounded-xl w-[160px]"
          />
        </div>
        <div className="space-y-2">
          <Label className="text-xs">To</Label>
          <Input
            type="date"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            className="rounded-xl w-[160px]"
          />
        </div>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <Calendar className="h-4 w-4" />
            Sessions ({data?.total ?? 0})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-20 w-full rounded-xl" />
              ))}
            </div>
          ) : sessions.length === 0 ? (
            <p className="text-muted-foreground py-12 text-center">No sessions in range.</p>
          ) : (
            <ul className="space-y-3">
              {sessions.map((s) => (
                <li key={s.id}>
                  <Link href={`/sessions/${s.id}`}>
                    <Card className="rounded-2xl shadow-sm transition hover:shadow-md">
                      <CardContent className="p-4 flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">
                            {new Date(s.started_at).toLocaleString()}
                          </p>
                          <p className="text-muted-foreground text-xs">
                            {s.mode} · {s.consent_state && Object.keys(s.consent_state).length > 0 ? "Consent set" : "Default consent"}
                          </p>
                          {s.summary_text && (
                            <p className="text-muted-foreground text-xs mt-1 line-clamp-1">
                              {s.summary_text}
                            </p>
                          )}
                        </div>
                        <Button variant="ghost" size="sm" className="rounded-xl">
                          View events
                        </Button>
                      </CardContent>
                    </Card>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
