"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

/** Ingest is now available from Graph view. Redirect so bookmarks still work. */
export default function IngestRedirectPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/graph");
  }, [router]);
  return (
    <div className="flex items-center justify-center min-h-[200px] text-muted-foreground text-sm">
      Redirecting to Graph view…
    </div>
  );
}
