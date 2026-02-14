"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Shield } from "lucide-react";

export function PolicyGateCard({
  withheld,
  couldShare,
}: {
  withheld?: string[];
  couldShare?: string[];
}) {
  const w = withheld?.length ?? 0;
  const c = couldShare?.length ?? 0;
  if (w === 0 && c === 0) return null;

  return (
    <Card className="rounded-2xl shadow-sm border-amber-200 bg-amber-50/50 dark:border-amber-900 dark:bg-amber-950/20">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Shield className="h-4 w-4" />
          Policy gate
        </CardTitle>
        <p className="text-muted-foreground text-sm">
          What was withheld due to consent and what could be shared if the elder opts in.
        </p>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {w > 0 && (
          <div>
            <p className="font-medium text-amber-800 dark:text-amber-200">Withheld</p>
            <ul className="list-disc list-inside text-muted-foreground">
              {(withheld ?? []).map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {c > 0 && (
          <div>
            <p className="font-medium text-green-800 dark:text-green-200">Could share if opted in</p>
            <ul className="list-disc list-inside text-muted-foreground">
              {(couldShare ?? []).map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
