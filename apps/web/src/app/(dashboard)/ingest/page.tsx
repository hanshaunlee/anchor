"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useIngestEventsMutation } from "@/hooks/use-api";
import { useAppStore } from "@/store/use-app-store";
import { Upload } from "lucide-react";

const PLACEHOLDER = `Paste a JSON array of event packets. Each event must include:
- session_id (UUID, must belong to your household)
- device_id (UUID, must belong to your household)
- ts (ISO8601)
- seq (number)
- event_type (string)
- payload (object)

Example:
[
  {
    "session_id": "...",
    "device_id": "...",
    "ts": "2025-02-14T12:00:00.000Z",
    "seq": 0,
    "event_type": "final_asr",
    "payload": { "text": "Hello", "confidence": 0.9 }
  }
]`;

export default function IngestPage() {
  const [jsonInput, setJsonInput] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);
  const demoMode = useAppStore((s) => s.demoMode);
  const mutation = useIngestEventsMutation();

  const handleSubmit = () => {
    setParseError(null);
    let events: unknown[];
    try {
      const parsed = JSON.parse(jsonInput.trim() || "[]");
      events = Array.isArray(parsed) ? parsed : [parsed];
    } catch (e) {
      setParseError(e instanceof Error ? e.message : "Invalid JSON");
      return;
    }
    if (events.length === 0) {
      setParseError("Events array is empty");
      return;
    }
    mutation.mutate(
      { events },
      {
        onSuccess: () => {
          setJsonInput("");
        },
        onError: (err) => {
          setParseError(err instanceof Error ? err.message : "Ingest failed");
        },
      }
    );
  };

  if (demoMode) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Upload className="h-6 w-6" />
          Ingest events
        </h1>
        <p className="text-muted-foreground text-sm">
          Ingest is disabled in demo mode. Turn off demo mode and sign in to upload event batches.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Upload className="h-6 w-6" />
          Ingest events
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Batch upload event packets. Sessions and devices must belong to your household. Re-uploading the same events (same session_id + seq) is safe and updates existing rows (idempotent).
        </p>
      </div>

      <Card className="rounded-2xl shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Event batch</CardTitle>
          <p className="text-muted-foreground text-sm">
            Paste a JSON array of events. Required fields: session_id, device_id, ts, seq, event_type, payload. Duplicate (session_id, seq) are upserted, not duplicated.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder={PLACEHOLDER}
            value={jsonInput}
            onChange={(e) => {
              setJsonInput(e.target.value);
              setParseError(null);
            }}
            className="min-h-[280px] rounded-xl font-mono text-sm"
          />
          {(parseError || mutation.isError) && (
            <p className="text-destructive text-sm">
              {mutation.error instanceof Error ? mutation.error.message : parseError}
            </p>
          )}
          {mutation.isSuccess && mutation.data && (
            <p className="text-green-600 dark:text-green-400 text-sm">
              Ingested {mutation.data.ingested} event(s). Session IDs: {mutation.data.session_ids?.length ?? 0}.
            </p>
          )}
          <Button
            className="rounded-xl"
            onClick={handleSubmit}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "Ingesting…" : "Ingest"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
