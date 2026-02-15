import { create } from "zustand";

export type DemoMode = boolean;

interface AppState {
  /** Supabase JWT for API calls */
  token: string | null;
  setToken: (t: string | null) => void;
  /** Demo mode: use fixtures, no real API */
  demoMode: boolean;
  setDemoMode: (v: boolean) => void;
  /** Scenario replay: currently playing */
  replayingScenarioId: string | null;
  setReplayingScenarioId: (id: string | null) => void;
  /** Judge mode: show model internals (calibrated_p, rule_score, etc.) on alert detail */
  judgeMode: boolean;
  setJudgeMode: (v: boolean) => void;
  /** Explain mode: show agent names, traces, artifact IDs, model health/drift/calibration (judge/dev) */
  explainMode: boolean;
  setExplainMode: (v: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  token: null,
  setToken: (token) => set({ token }),
  demoMode: process.env.NEXT_PUBLIC_DEMO_MODE === "true",
  setDemoMode: (demoMode) => set({ demoMode }),
  replayingScenarioId: null,
  setReplayingScenarioId: (replayingScenarioId) => set({ replayingScenarioId }),
  judgeMode: false,
  setJudgeMode: (judgeMode) => set({ judgeMode }),
  explainMode: false,
  setExplainMode: (explainMode) => set({ explainMode }),
}));
