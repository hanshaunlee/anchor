"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useCapabilitiesMe, usePatchCapabilitiesMutation, useHouseholdMe } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Settings } from "lucide-react";

export default function SettingsPage() {
  const { data: me } = useHouseholdMe();
  const { data: cap, isLoading } = useCapabilitiesMe();
  const patchMutation = usePatchCapabilitiesMutation();
  const canEdit = me?.role === "caregiver" || me?.role === "admin";

  if (isLoading || !cap) {
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
