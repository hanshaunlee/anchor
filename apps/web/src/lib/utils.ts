import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Human-facing risk tier from numeric score (0–1). Family mode shows this instead of raw %. */
export function scoreToRiskTier(score: number): "Low" | "Medium" | "High" {
  if (score >= 0.6) return "High"
  if (score >= 0.3) return "Medium"
  return "Low"
}
