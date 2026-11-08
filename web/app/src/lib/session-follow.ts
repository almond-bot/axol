/**
 * Several browsers can watch one robot. A run started from any of them must
 * show up on all of them without a reload, so the control panel follows the
 * host's live operation rather than only the one it started itself.
 */
import type { SessionInfo } from "./supervisor"

export function isLiveSession(session: SessionInfo): boolean {
  return (
    session.status === "starting" || session.status === "running" || session.status === "stopping"
  )
}

/**
 * The live operation from the host's session list that this panel is not
 * showing yet, or null when there is none or the panel already shows it.
 */
export function operationToFollow(
  sessions: readonly SessionInfo[],
  operationIds: ReadonlySet<string>,
  shownSessionId: string | null
): SessionInfo | null {
  const live = sessions.find((s) => isLiveSession(s) && operationIds.has(s.command))
  if (!live || live.id === shownSessionId) return null
  return live
}
