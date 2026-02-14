"use client";

import { useCallback, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getWsUrl } from "@/lib/api";
import { RiskSignalWsMessageSchema, type RiskSignalCard } from "@/lib/api/schemas";

const WS_PATH = "/ws/risk_signals";

/**
 * Subscribe to live risk signals over WebSocket; merge into TanStack Query cache.
 * Fallback to polling is handled by refetchInterval in useRiskSignals when WS fails.
 */
export function useRiskSignalStream(enabled: boolean) {
  const queryClient = useQueryClient();

  const handleMessage = useCallback(
    (raw: string) => {
      try {
        const msg = JSON.parse(raw);
        const parsed = RiskSignalWsMessageSchema.safeParse(msg);
        if (!parsed.success) return;
        const { id, ts, signal_type, severity, score } = parsed.data;
        const card: RiskSignalCard = {
          id,
          ts,
          signal_type,
          severity,
          score,
          status: "open",
          summary: null,
        };
        queryClient.setQueryData(
          ["risk_signals"],
          (old: { signals: RiskSignalCard[]; total: number } | undefined) => {
            if (!old) return { signals: [card], total: 1 };
            const exists = old.signals.some((s) => s.id === id);
            if (exists) return old;
            return {
              signals: [card, ...old.signals],
              total: old.total + 1,
            };
          }
        );
      } catch {
        // ignore invalid messages
      }
    },
    [queryClient]
  );

  useEffect(() => {
    if (!enabled) return;
    const url = getWsUrl(WS_PATH);
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(url);
      ws.onmessage = (e) => handleMessage(e.data);
    } catch {
      // fallback to polling in query
    }
    return () => {
      ws?.close();
    };
  }, [enabled, handleMessage]);
}
