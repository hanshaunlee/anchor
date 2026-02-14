"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";

export default function OnboardPage() {
  const [displayName, setDisplayName] = useState("");
  const [householdName, setHouseholdName] = useState("My Household");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await api.postOnboard({
        display_name: displayName || undefined,
        household_name: householdName || undefined,
      });
      router.push("/dashboard");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Setup failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4 bg-anchor-warm">
      <Card className="w-full max-w-md rounded-2xl shadow-lg bg-card">
        <CardHeader className="space-y-1">
          <div className="flex justify-center mb-2">
            <img src="/logo.png" alt="Anchor" className="h-16 w-auto max-h-20 sm:h-20 sm:max-h-24 object-contain" />
          </div>
          <CardTitle className="text-2xl text-center">Set up your household</CardTitle>
          <CardDescription>
            You’re signed in. Create your household to start using Anchor.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
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
              <Label htmlFor="householdName">Household name</Label>
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
              {loading ? "Creating…" : "Create household"}
            </Button>
          </form>
          <p className="mt-4 text-center text-muted-foreground text-sm">
            <Link href="/logout" className="underline hover:text-foreground">
              Sign out
            </Link>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
