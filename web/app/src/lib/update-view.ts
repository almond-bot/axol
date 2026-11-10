import type { UpdatePhase } from "@/lib/supervisor"

/** Human-readable label for the current update step (shown while updating). */
export const PHASE_LABEL: Record<UpdatePhase, string> = {
  upgrading: "Upgrading…",
  provisioning: "Installing packages…",
  restarting: "Restarting…",
  rebooting: "Rebooting host…",
}

/** A commit shortened to its first seven characters. */
export function shortCommit(commit: string | null | undefined): string {
  return commit ? commit.slice(0, 7) : "unknown"
}
