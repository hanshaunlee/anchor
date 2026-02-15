"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type ReportMiniCardProps = {
  kind: string;
  description: string;
  lastRunAt?: string | null;
  lastRunId?: string | null;
  summary?: string | null;
  status?: string | null;
  href: string;
  buttonLabel: string;
  icon?: React.ReactNode;
  className?: string;
};

export function ReportMiniCard({
  kind,
  description,
  lastRunAt,
  summary,
  href,
  buttonLabel,
  icon,
  className,
}: ReportMiniCardProps) {
  return (
    <Card className={cn("rounded-2xl border-border", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          {icon}
          {kind}
        </CardTitle>
        <p className="text-muted-foreground text-sm">{description}</p>
        {lastRunAt && (
          <p className="text-xs text-muted-foreground">
            Last run: {formatDate(lastRunAt)}
          </p>
        )}
        {summary && (
          <p className="text-xs text-muted-foreground line-clamp-2">{summary}</p>
        )}
      </CardHeader>
      <CardContent>
        <Link href={href}>
          <Button variant="outline" size="sm" className="rounded-xl">
            {buttonLabel}
          </Button>
        </Link>
      </CardContent>
    </Card>
  );
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
