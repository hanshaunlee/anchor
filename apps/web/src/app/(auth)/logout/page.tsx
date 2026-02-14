"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { useAppStore } from "@/store/use-app-store";

export default function LogoutPage() {
  const router = useRouter();
  const setToken = useAppStore((s) => s.setToken);

  useEffect(() => {
    setToken(null);
    if (typeof window !== "undefined") window.__anchor_token = undefined;
    supabase?.auth.signOut().finally(() => {
      router.push("/");
      router.refresh();
    });
  }, [router, setToken]);

  return (
    <div className="flex min-h-screen items-center justify-center">
      <p className="text-muted-foreground">Signing out…</p>
    </div>
  );
}
