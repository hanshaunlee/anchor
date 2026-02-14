"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRiskSignals, useHouseholdMe, useOutreachActions } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { motion } from "framer-motion";

export default function ElderPage() {
  const { data: me } = useHouseholdMe();
  const { data: signalsData } = useRiskSignals({ limit: 5 });
  const { data: outreachData } = useOutreachActions({ limit: 10 });
  const signals = signalsData?.signals ?? [];
  const openAlerts = signals.filter((s) => s.status === "open");
  const hasConcern = openAlerts.length > 0;
  const latestOutreach = (outreachData?.actions ?? [])[0] as Record<string, unknown> | undefined;
  const elderSafeMessage =
    latestOutreach?.payload && typeof latestOutreach.payload === "object" && "elder_safe_message" in latestOutreach.payload
      ? (latestOutreach.payload as { elder_safe_message?: string }).elder_safe_message
      : null;
  const caregiverNotified = !!latestOutreach && (latestOutreach.status === "sent" || latestOutreach.status === "delivered");

  return (
    <div className="max-w-lg mx-auto space-y-8 py-8 px-4">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-semibold tracking-tight">
          {me?.display_name ? `Hi, ${me.display_name}` : "Hello"}
        </h1>
        <p className="text-muted-foreground text-lg mt-2">Here’s your summary for today.</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="rounded-2xl shadow-sm text-lg">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl">Today</CardTitle>
          </CardHeader>
          <CardContent>
            {signalsData === undefined ? (
              <Skeleton className="h-12 w-full rounded-xl" />
            ) : hasConcern ? (
              <p className="leading-relaxed">
                We noticed something that might need attention. We recommend sharing this with your caregiver.
              </p>
            ) : (
              <p className="leading-relaxed text-muted-foreground">
                Everything looks normal. No new alerts.
              </p>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {hasConcern && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Card className="rounded-2xl shadow-sm border-amber-200 bg-amber-50/50 dark:border-amber-900 dark:bg-amber-950/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">Recommendation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <p className="text-sm leading-relaxed font-medium">
                We&apos;re in high-risk mode. Don&apos;t share codes. Your caregiver has been notified.
              </p>
              {elderSafeMessage ? (
                <p className="text-sm leading-relaxed text-muted-foreground">{elderSafeMessage}</p>
              ) : (
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Consider talking to your caregiver about recent contacts or requests. No raw conversation text is shared unless you allow it.
                </p>
              )}
              {caregiverNotified && (
                <p className="text-xs text-muted-foreground font-medium">Caregiver notified.</p>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex items-center justify-between rounded-2xl border border-border p-4"
      >
        <Label htmlFor="share-toggle" className="text-base cursor-pointer">
          Share with caregiver
        </Label>
        <Switch id="share-toggle" className="data-[state=checked]:bg-primary" />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <Card className="rounded-2xl shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-xl">Recent interactions</CardTitle>
          </CardHeader>
          <CardContent>
            {signals.length === 0 ? (
              <p className="text-muted-foreground text-sm">No recent interactions to show.</p>
            ) : (
              <ul className="space-y-2 text-sm">
                {signals.slice(0, 3).map((s) => (
                  <li key={s.id} className="flex justify-between gap-2">
                    <span className="text-muted-foreground truncate">
                      {s.summary ?? `${s.signal_type} (no details)`}
                    </span>
                    <span className="text-muted-foreground text-xs shrink-0">
                      {new Date(s.ts).toLocaleDateString()}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
