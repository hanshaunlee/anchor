"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { DashboardNav } from "@/components/dashboard-nav";
import { useHouseholdMe } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export default function DashboardLayout({
  children,
}: { children: React.ReactNode }) {
  const router = useRouter();
  const { data: me, isLoading, isError, error } = useHouseholdMe();
  const demoMode = useAppStore((s) => s.demoMode);
  const setDemoMode = useAppStore((s) => s.setDemoMode);

  useEffect(() => {
    if (demoMode) return;
    if (isLoading) return;
    if (isError && error && (String(error).includes("404") || String(error).toLowerCase().includes("user not found") || String(error).toLowerCase().includes("not onboarded"))) {
      router.replace("/onboard");
    }
  }, [demoMode, isLoading, isError, error, router]);

  return (
    <div className="flex min-h-screen">
      <aside className="w-56 border-r border-border bg-card p-4 flex flex-col shrink-0">
        <Link href="/dashboard" className="mb-6 block">
          <span className="text-lg font-semibold">Anchor</span>
        </Link>
        <DashboardNav />
        <div className="mt-auto pt-4 border-t border-border space-y-2">
          {isLoading ? (
            <Skeleton className="h-8 w-full rounded-xl" />
          ) : (
            <p className="text-muted-foreground text-xs truncate">
              {me?.display_name || me?.name || "—"}
            </p>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start rounded-xl text-xs"
            onClick={() => setDemoMode(!demoMode)}
          >
            {demoMode ? "Demo mode ON" : "Demo mode OFF"}
          </Button>
          <a href="/logout" className="block">
            <Button variant="ghost" size="sm" className="w-full justify-start rounded-xl text-xs text-muted-foreground">
              Sign out
            </Button>
          </a>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-6 md:p-8">{children}</main>
    </div>
  );
}
