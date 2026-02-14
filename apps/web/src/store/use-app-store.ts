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
}

export const useAppStore = create<AppState>((set) => ({
  token: null,
  setToken: (token) => set({ token }),
  demoMode: process.env.NEXT_PUBLIC_DEMO_MODE === "true",
  setDemoMode: (demoMode) => set({ demoMode }),
  replayingScenarioId: null,
  setReplayingScenarioId: (replayingScenarioId) => set({ replayingScenarioId }),
}));
