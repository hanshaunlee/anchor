"use client";

import { useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle2 } from "lucide-react";

export default function SignUpSuccessPage() {
  useEffect(() => {
    const t = setTimeout(() => {
      window.location.href = "/dashboard";
    }, 5000);
    return () => clearTimeout(t);
  }, []);

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="w-full max-w-md rounded-2xl shadow-lg border-green-200 dark:border-green-900/50 bg-green-50/30 dark:bg-green-950/20">
        <CardHeader className="space-y-1 text-center pb-2">
          <div className="flex justify-center">
            <CheckCircle2 className="h-14 w-14 text-green-600 dark:text-green-400" aria-hidden />
          </div>
          <CardTitle className="text-2xl">Account created successfully</CardTitle>
          <CardDescription>
            Your household is set up. You can start using Anchor now.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-center text-muted-foreground">
            You’ll be taken to the dashboard in a few seconds, or go now:
          </p>
          <Link href="/dashboard" className="block">
            <Button className="w-full rounded-xl" size="lg">
              Go to dashboard
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
