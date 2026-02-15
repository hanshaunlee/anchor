"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, ShieldAlert, ClipboardCopy, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export type PlaybookTask = {
  id: string;
  task_type: string;
  status: string;
  details: Record<string, unknown>;
};

export type OutreachAction = Record<string, unknown>;

export type DecisionPanelProps = {
  /** Recommended action steps */
  steps: string[];
  /** Notify caregiver: allowed by policy */
  canNotifyCaregiver: boolean;
  demoMode: boolean;
  /** Outreach mutation: preview (dry_run: true) then send (dry_run: false) */
  onPreviewMessage: () => void;
  onConfirmSend: () => void;
  previewPending: boolean;
  sendPending: boolean;
  /** Preview state from parent */
  outreachPreview: { caregiver_message?: string; elder_safe_message?: string } | null;
  onClearPreview: () => void;
  /** Past outreach for this signal */
  outreachActions: OutreachAction[];
  outreachError: Error | null;
  /** Incident response: playbook + run */
  playbook: { id: string; tasks: PlaybookTask[] } | null;
  playbookLoading: boolean;
  onRunIncidentResponse: () => void;
  incidentResponsePending: boolean;
  incidentResponseError: Error | null;
  onCompleteTask: (taskId: string) => void;
  completeTaskPending: boolean;
  /** Bank lock card capability */
  hasLockCardCap: boolean;
  capabilitiesNote?: string | null;
};

export function DecisionPanel({
  steps,
  canNotifyCaregiver,
  demoMode,
  onPreviewMessage,
  onConfirmSend,
  previewPending,
  sendPending,
  outreachPreview,
  onClearPreview,
  outreachActions,
  outreachError,
  playbook,
  playbookLoading,
  onRunIncidentResponse,
  incidentResponsePending,
  incidentResponseError,
  onCompleteTask,
  completeTaskPending,
  hasLockCardCap,
  capabilitiesNote,
}: DecisionPanelProps) {
  const [copiedScript, setCopiedScript] = useState(false);

  return (
    <Card className="rounded-2xl shadow-sm border-primary/20">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold">Recommended actions</CardTitle>
        <p className="text-muted-foreground text-sm">
          Immediate actions, escalation status, and incident response
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 1. Immediate actions */}
        <section>
          <h3 className="text-sm font-medium mb-2">1. Immediate actions</h3>
          {Array.isArray(steps) && steps.length > 0 ? (
            <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
              {steps.map((s, i) => (
                <li key={i}>{String(s)}</li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">No specific steps.</p>
          )}
        </section>

        {/* 2. Notify caregiver */}
        {canNotifyCaregiver && !demoMode && (
          <section>
            <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
              <Bell className="h-4 w-4" />
              Notify caregiver
            </h3>
            {outreachError && (
              <p className="text-destructive text-sm rounded-lg bg-destructive/10 px-3 py-2 mb-2">
                {outreachError.message}
              </p>
            )}
            {!outreachPreview && (
              <Button
                variant="outline"
                size="sm"
                className="rounded-xl"
                disabled={previewPending}
                onClick={onPreviewMessage}
              >
                {previewPending ? "Loading…" : "Preview message"}
              </Button>
            )}
            {outreachPreview && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">Caregiver will see:</p>
                <div className="rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm whitespace-pre-line">
                  {outreachPreview.caregiver_message}
                </div>
                <p className="text-xs text-muted-foreground">{outreachPreview.elder_safe_message}</p>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    className="rounded-xl"
                    disabled={sendPending}
                    onClick={onConfirmSend}
                  >
                    {sendPending ? "Sending…" : "Confirm send"}
                  </Button>
                  <Button variant="ghost" size="sm" className="rounded-xl" onClick={onClearPreview}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </section>
        )}

        {/* 3. Escalation status */}
        {outreachActions.length > 0 && (
          <section>
            <h3 className="text-sm font-medium mb-2">2. Escalation status</h3>
            <p className="text-xs text-muted-foreground mb-1">Has caregiver been notified?</p>
            <ul className="space-y-1 text-sm">
              {outreachActions.map((a: OutreachAction, i: number) => {
                const status = (a.status as string) ?? "—";
                const err = a.error as string | undefined;
                const isSuppressed =
                  status === "suppressed" &&
                  (String(err).includes("consent_allow_outbound") ?? String(err).includes("consent"));
                return (
                  <li key={i} className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">{status}</span>
                    {a.created_at && (
                      <span className="text-muted-foreground">
                        Created {new Date(String(a.created_at)).toLocaleString()}
                      </span>
                    )}
                    {a.sent_at && (
                      <span className="text-muted-foreground">
                        Sent {new Date(String(a.sent_at)).toLocaleString()}
                      </span>
                    )}
                    {isSuppressed && (
                      <span className="block text-muted-foreground text-xs w-full">
                        Outbound contact is off in settings.
                      </span>
                    )}
                    {err && !isSuppressed && (
                      <span className="block text-destructive text-xs w-full">Error: {String(err)}</span>
                    )}
                  </li>
                );
              })}
            </ul>
          </section>
        )}

        {/* 4. Incident response */}
        <section>
          <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
            <ShieldAlert className="h-4 w-4" />
            3. Incident response
          </h3>
          {capabilitiesNote && !hasLockCardCap && (
            <p className="text-xs text-amber-700 dark:text-amber-400 rounded-lg bg-amber-500/10 px-3 py-2 mb-2">
              {capabilitiesNote}
            </p>
          )}
          {incidentResponseError && (
            <p className="text-destructive text-sm rounded-lg bg-destructive/10 px-3 py-2 mb-2">
              {incidentResponseError.message}
            </p>
          )}
          {!playbookLoading && !playbook && canNotifyCaregiver && !demoMode && (
            <Button
              size="sm"
              className="rounded-xl"
              disabled={incidentResponsePending}
              onClick={onRunIncidentResponse}
            >
              {incidentResponsePending ? "Running…" : "Run Incident Response"}
            </Button>
          )}
          {!playbookLoading && playbook && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">
                Playbook {playbook.id?.slice(0, 8)}… · {playbook.tasks?.length ?? 0} tasks
              </p>
              <ul className="space-y-2">
                {(playbook.tasks ?? []).map((task) => (
                  <li
                    key={task.id}
                    className="flex items-center gap-3 rounded-xl border border-border px-3 py-2 text-sm"
                  >
                    <span className="font-medium capitalize min-w-0 flex-1">
                      {task.task_type.replace(/_/g, " ")}
                    </span>
                    <Badge
                      variant={task.status === "done" ? "default" : "secondary"}
                      className="rounded-lg shrink-0 w-14 justify-center"
                    >
                      {task.status}
                    </Badge>
                    <div className="flex items-center gap-2 shrink-0">
                      {task.task_type === "call_bank" && task.details?.call_script && task.status !== "done" && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="rounded-lg"
                          onClick={() => {
                            navigator.clipboard.writeText(String(task.details.call_script));
                            setCopiedScript(true);
                            setTimeout(() => setCopiedScript(false), 2000);
                          }}
                        >
                          <ClipboardCopy className="h-4 w-4 mr-1" />
                          {copiedScript ? "Copied" : "Copy script"}
                        </Button>
                      )}
                      {task.status !== "done" && task.status !== "blocked" && (
                        <Button
                          size="sm"
                          variant="outline"
                          className="rounded-lg"
                          disabled={completeTaskPending}
                          onClick={() => onCompleteTask(task.id)}
                        >
                          <CheckCircle className="h-4 w-4 mr-1" />
                          Mark done
                        </Button>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </CardContent>
    </Card>
  );
}
