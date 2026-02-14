"use client";

import { useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-[40vh] flex-col items-center justify-center gap-4 p-8">
      <h2 className="text-lg font-semibold">Something went wrong</h2>
      <p className="text-muted-foreground text-center text-sm max-w-md">
        {error.message || "An unexpected error occurred."}
      </p>
      <div className="flex gap-3">
        <Button onClick={reset} variant="default" className="rounded-xl">
          Try again
        </Button>
        <Link href="/dashboard">
          <Button variant="outline" className="rounded-xl">
            Back to dashboard
          </Button>
        </Link>
      </div>
    </div>
  );
}
