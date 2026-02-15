"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useCapabilitiesMe, usePatchCapabilitiesMutation, useHouseholdMe, useHouseholdConsent, usePatchHouseholdConsentMutation } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Settings, AlertCircle, RefreshCw } from "lucide-react";

export default function SettingsPage() {
  const { data: me } = useHouseholdMe();
  const { data: cap, isLoading, isError, error, refetch } = useCapabilitiesMe();
  const { data: consent, isLoading: consentLoading } = useHouseholdConsent();
  const patchMutation = usePatchCapabilitiesMutation();
  const patchConsentMutation = usePatchHouseholdConsentMutation();
  const canEdit = me?.role === "caregiver" || me?.role === "admin";

  if (isLoading && !cap) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  if (isError && !cap) {
    const errStr = String(error).toLowerCase();
    const notOnboarded =
      String(error).includes("404") || errStr.includes("not onboarded");
    const unauthorized =
      String(error).includes("401") || errStr.includes("not authenticated") || errStr.includes("unauthorized");
    const forbidden =
      String(error).includes("403") || errStr.includes("forbidden") || errStr.includes("only caregivers");
    const backendUnavailable =
      String(error).includes("503") || errStr.includes("not configured") || errStr.includes("supabase");
    const networkError =
      errStr.includes("failed to fetch") || errStr.includes("networkerror") || errStr.includes("fetch");

    let message: string;
    if (notOnboarded) {
      message = "Complete onboarding first to manage household capabilities.";
    } else if (unauthorized) {
      message = "Please sign in again. If you just signed in, try refreshing the page.";
    } else if (forbidden) {
      message = "Only caregivers or admins can view and edit settings.";
    } else if (backendUnavailable) {
      message = "Backend is not fully configured. Check that Supabase and the API are set up.";
    } else if (networkError) {
      const apiBase =
        typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_BASE_URL
          ? process.env.NEXT_PUBLIC_API_BASE_URL.trim()
          : "http://localhost:8000";
      message = `Cannot reach the API. Ensure it is running (e.g. at ${apiBase}). You can use Demo mode in the sidebar to try the app without the API.`;
    } else {
      message = "Check your connection and try again.";
    }

    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Settings className="h-7 w-7" />
          Settings
        </h1>
        <Card className="rounded-2xl border-destructive/50 bg-destructive/5">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div className="space-y-2">
                <p className="font-medium">Unable to load settings</p>
                <p className="text-sm text-muted-foreground">{message}</p>
                <div className="flex gap-2 mt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="rounded-lg"
                    onClick={() => refetch()}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Retry
                  </Button>
                  {unauthorized && (
                    <Button variant="ghost" size="sm" className="rounded-lg" asChild>
                      <a href="/logout">Sign out</a>
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!cap) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }

  const bankCaps = (cap.bank_control_capabilities || {}) as Record<string, boolean>;
  const update = (key: string, value: boolean) => {
    if (!canEdit) return;
    patchMutation.mutate({
      bank_control_capabilities: { ...bankCaps, [key]: value },
    });
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Settings className="h-7 w-7" />
          Settings
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Household capabilities and connected providers. Action buttons in alerts respect these.
        </p>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Capabilities</CardTitle>
          <p className="text-muted-foreground text-sm">
            When a capability is off, the UI shows &quot;Not available&quot; and we provide scripts/playbooks instead.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between rounded-xl border border-border px-4 py-3">
            <Label htmlFor="notify-sms" className="cursor-pointer">Notify via SMS</Label>
            <Switch
              id="notify-sms"
              checked={cap.notify_sms_enabled}
              disabled={!canEdit}
              onCheckedChange={(v) => patchMutation.mutate({ notify_sms_enabled: v })}
            />
          </div>
          <div className="flex items-center justify-between rounded-xl border border-border px-4 py-3">
            <Label htmlFor="notify-email" className="cursor-pointer">Notify via email</Label>
            <Switch
              id="notify-email"
              checked={cap.notify_email_enabled}
              disabled={!canEdit}
              onCheckedChange={(v) => patchMutation.mutate({ notify_email_enabled: v })}
            />
          </div>
          <div className="flex items-center justify-between rounded-xl border border-border px-4 py-3">
            <Label htmlFor="device-push" className="cursor-pointer">Device policy push (high-risk mode)</Label>
            <Switch
              id="device-push"
              checked={cap.device_policy_push_enabled}
              disabled={!canEdit}
              onCheckedChange={(v) => patchMutation.mutate({ device_policy_push_enabled: v })}
            />
          </div>
        </CardContent>
      </Card>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Outbound contact</CardTitle>
          <p className="text-muted-foreground text-sm">
            When on, alerts can send messages to the caregiver (SMS/email). When off, preview and send are suppressed.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {consentLoading && !consent ? (
            <Skeleton className="h-12 w-full rounded-xl" />
          ) : consent ? (
            <div className="flex items-center justify-between rounded-xl border border-border px-4 py-3">
              <Label htmlFor="allow-outbound" className="cursor-pointer">Allow outbound contact</Label>
              <Switch
                id="allow-outbound"
                checked={consent.allow_outbound_contact}
                disabled={!canEdit}
                onCheckedChange={(v) => patchConsentMutation.mutate({ allow_outbound_contact: v })}
              />
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Bank control (demo)</CardTitle>
          <p className="text-muted-foreground text-sm">
            Your bank integration does not support lock card by default; we prepare the call script instead.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Connected provider: <span className="font-medium">{cap.bank_data_connector || "none"}</span>
          </p>
          <div className="flex items-center justify-between rounded-xl border border-border px-4 py-3">
            <Label htmlFor="lock-card" className="cursor-pointer">Lock card (if supported)</Label>
            <Switch
              id="lock-card"
              checked={bankCaps.lock_card === true}
              disabled={!canEdit}
              onCheckedChange={(v) => update("lock_card", v)}
            />
          </div>
          <div className="flex items-center justify-between rounded-xl border border-border px-4 py-3">
            <Label htmlFor="enable-alerts" className="cursor-pointer">Enable alerts</Label>
            <Switch
              id="enable-alerts"
              checked={bankCaps.enable_alerts !== false}
              disabled={!canEdit}
              onCheckedChange={(v) => update("enable_alerts", v)}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
