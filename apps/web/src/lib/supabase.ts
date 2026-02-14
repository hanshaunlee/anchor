import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

export const supabase = url && anon ? createClient(url, anon) : null;

declare global {
  interface Window {
    __anchor_token?: string;
  }
}

export function setAnchorToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) window.__anchor_token = token;
  else delete window.__anchor_token;
}
