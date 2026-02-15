"use client";

import { AlertCircle } from "lucide-react";

export function ConsentGateBanner({
  message,
  missingKeys,
}: {
  message: string;
  missingKeys?: string[];
}) {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50/50 dark:border-amber-900 dark:bg-amber-950/20 px-4 py-3 flex items-start gap-3">
      <AlertCircle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
      <div className="text-sm">
        <p className="font-medium text-amber-800 dark:text-amber-200">{message}</p>
        {missingKeys && missingKeys.length > 0 && (
          <p className="text-muted-foreground mt-1">
            Missing or disabled: {missingKeys.join(", ")}. Update consent or household settings to enable.
          </p>
        )}
      </div>
    </div>
  );
}
