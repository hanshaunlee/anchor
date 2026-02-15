/**
 * Call after investigation run, alert refresh, or any agent run that changes
 * graph/evidence/watchlists/rings. Invalidates risk signals, similar incidents,
 * watchlists, rings, reports, protection overview/summary, graph evidence, outreach.
 * Use with queryClient from useQueryClient().
 */
export { invalidateAfterInvestigation } from "@/hooks/use-api";
