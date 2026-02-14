"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { supabase } from "@/lib/supabase";
import { setAnchorToken } from "@/lib/supabase";
import { api } from "@/lib/api";

export default function SignUpPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [householdName, setHouseholdName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [needConfirm, setNeedConfirm] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setNeedConfirm(false);
    setLoading(true);
    if (!supabase) {
      setError("Auth not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY.");
      setLoading(false);
      return;
    }
    const { data, error: signUpError } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { display_name: displayName || undefined },
      },
    });
    if (signUpError) {
      const msg = signUpError.message || "Sign up failed";
      const code = (signUpError as { code?: string }).code;
      setError(code ? `${msg} (${code})` : msg);
      setLoading(false);
      return;
    }
    if (data.session) {
      setAnchorToken(data.session.access_token);
      try {
        await api.postOnboard({
          display_name: displayName || undefined,
          household_name: householdName || undefined,
        });
      } catch (onboardErr) {
        setError(onboardErr instanceof Error ? onboardErr.message : "Setup failed");
        setLoading(false);
        return;
      }
      router.push("/signup/success");
      router.refresh();
    } else {
      setNeedConfirm(true);
    }
    setLoading(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4 bg-[#F8F8F5]">
      <Card className="w-full max-w-md rounded-2xl shadow-lg bg-card">
        <CardHeader className="space-y-1">
          <div className="flex justify-center mb-2">
            <img src="/logo.png" alt="Anchor" className="h-20 w-auto sm:h-24" />
          </div>
          <CardTitle className="text-2xl text-center">Create an account</CardTitle>
          <CardDescription>
            Sign up for Anchor. You’ll get your own household and can invite others later.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {needConfirm ? (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Check your email to confirm your account. Then sign in to continue.
              </p>
              <Link href="/login">
                <Button variant="outline" className="w-full rounded-xl">
                  Go to sign in
                </Button>
              </Link>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="rounded-xl"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="rounded-xl"
                  required
                  minLength={6}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="displayName">Display name (optional)</Label>
                <Input
                  id="displayName"
                  type="text"
                  placeholder="Your name"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="rounded-xl"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="householdName">Household name (optional)</Label>
                <Input
                  id="householdName"
                  type="text"
                  placeholder="My Household"
                  value={householdName}
                  onChange={(e) => setHouseholdName(e.target.value)}
                  className="rounded-xl"
                />
              </div>
              {error && (
                <p className="text-destructive text-sm">{error}</p>
              )}
              <Button type="submit" className="w-full rounded-xl" size="lg" disabled={loading}>
                {loading ? "Creating account…" : "Create account"}
              </Button>
            </form>
          )}
          {!needConfirm && (
            <p className="mt-4 text-center text-muted-foreground text-sm">
              Already have an account?{" "}
              <Link href="/login" className="underline hover:text-foreground">
                Sign in
              </Link>
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
