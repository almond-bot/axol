/**
 * Most-recently-connected Axol hosts, offered as a dropdown in the setup
 * dialog. Stored newest first in localStorage; a host is recorded only once a
 * connection to it succeeds, so typos never make it into the list.
 */

export const HOST_HISTORY_STORAGE = "axolHostHistory"
export const HOST_HISTORY_LIMIT = 5

export function loadHostHistory(): string[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(HOST_HISTORY_STORAGE) ?? "[]")
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((h): h is string => typeof h === "string" && h.trim() !== "")
      .slice(0, HOST_HISTORY_LIMIT)
  } catch {
    return []
  }
}

/** Move `host` to the front (deduped case-insensitively) and persist; returns the new list. */
export function recordHost(host: string): string[] {
  const trimmed = host.trim()
  const current = loadHostHistory()
  if (!trimmed) return current
  const next = [trimmed, ...current.filter((h) => h.toLowerCase() !== trimmed.toLowerCase())].slice(
    0,
    HOST_HISTORY_LIMIT
  )
  try {
    localStorage.setItem(HOST_HISTORY_STORAGE, JSON.stringify(next))
  } catch {
    // ignore storage failures
  }
  return next
}
