"use client";

import { useEffect } from "react";
import { useAppStore } from "@/store/use-app-store";
import { supabase, setAnchorToken } from "@/lib/supabase";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const setToken = useAppStore((s) => s.setToken);

  useEffect(() => {
    if (!supabase) return;
    const setSession = (token: string | null) => {
      setToken(token);
      setAnchorToken(token);
    };
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session?.access_token ?? null);
    });
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session?.access_token ?? null);
    });
    return () => subscription.unsubscribe();
  }, [setToken]);

  return <>{children}</>;
}
