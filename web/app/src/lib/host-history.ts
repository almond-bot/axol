/**
 * Most-recently-connected hosts, offered as a dropdown on the host field of
 * the control panel's setup dialog and the VR app's connect card. Stored
 * newest first in localStorage; a host is recorded only once a connection to
 * it succeeds, so typos never make it into the list. Each app keeps its own
 * list (the control panel and the teleop server can live on different hosts).
 */

export const HOST_HISTORY_STORAGE = "axolHostHistory"
export const VR_HOST_HISTORY_STORAGE = "vrHostHistory"
export const HOST_HISTORY_LIMIT = 5

export function loadHostHistory(storageKey = HOST_HISTORY_STORAGE): string[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(storageKey) ?? "[]")
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((h): h is string => typeof h === "string" && h.trim() !== "")
      .slice(0, HOST_HISTORY_LIMIT)
  } catch {
    return []
  }
}

/** Move `host` to the front (deduped case-insensitively) and persist; returns the new list. */
export function recordHost(host: string, storageKey = HOST_HISTORY_STORAGE): string[] {
  const trimmed = host.trim()
  const current = loadHostHistory(storageKey)
  if (!trimmed) return current
  const next = [trimmed, ...current.filter((h) => h.toLowerCase() !== trimmed.toLowerCase())].slice(
    0,
    HOST_HISTORY_LIMIT
  )
  try {
    localStorage.setItem(storageKey, JSON.stringify(next))
  } catch {
    // ignore storage failures
  }
  return next
}
