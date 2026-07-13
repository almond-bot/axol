import type { ServerInfo } from "@/lib/supervisor"
import { servedByBackend } from "@/lib/supervisor"

/** What the UI/backend skew check found (see {@link versionMismatch}). */
export interface VersionMismatch {
  /** The mismatched bundle is the one the backend itself serves (stale dist). */
  local: boolean
  /** The backend is a dev checkout (any commit), not a tag-pinned install. */
  serverDev: boolean
  /** The backend is on an older release than this UI was built for. */
  serverOlder: boolean
  uiCommit: string
  serverCommit: string
  uiVersion: string | null
  serverVersion: string | null
}

/** Numeric dotted-version comparison; null when either side isn't parseable. */
function olderThan(a: string, b: string): boolean | null {
  const pa = a.split(".").map(Number)
  const pb = b.split(".").map(Number)
  if (pa.some(Number.isNaN) || pb.some(Number.isNaN)) return null
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const da = pa[i] ?? 0
    const db = pb[i] ?? 0
    if (da !== db) return da < db
  }
  return false
}

/**
 * Decide whether this bundle and the backend it is talking to are meaningfully
 * out of sync. Compares the commit baked in at build time against the
 * backend's commit (/api/info), so it works on forks too — nothing here
 * references the upstream repository.
 *
 * Commits are compared directly, with one exception: a hosted bundle
 * (axol.almond.bot) talking to a *release install*. Release installs only
 * ever sit on release-tag commits while the hosted bundle is rebuilt from
 * every push, so their commits legitimately differ between releases; only a
 * release *version* difference is reported there to keep the warning
 * meaningful. Dev-checkout backends can be on any commit, so any difference
 * against them is real skew and is always reported — as is any difference
 * with the bundle the backend itself serves (a stale web/app/dist).
 */
export function versionMismatch(info: ServerInfo | null): VersionMismatch | null {
  const uiCommit = __AXOL_BUILD_COMMIT__
  const uiVersion = __AXOL_BUILD_VERSION__
  const serverCommit = info?.commit ?? null
  const serverVersion = info?.version ?? null
  if (!uiCommit || !serverCommit || uiCommit === serverCommit) return null
  const local = servedByBackend()
  const serverDev = !(info?.releaseInstall ?? false)
  const sameVersion = uiVersion !== null && uiVersion === serverVersion
  if (!local && !serverDev && sameVersion) return null
  const serverOlder =
    uiVersion !== null && serverVersion !== null
      ? (olderThan(serverVersion, uiVersion) ?? false)
      : false
  return { local, serverDev, serverOlder, uiCommit, serverCommit, uiVersion, serverVersion }
}
