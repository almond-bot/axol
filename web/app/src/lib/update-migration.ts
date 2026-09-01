/**
 * Releases before this one update themselves with the legacy, destructive
 * updater. That path cannot preserve custom CUDA Torch, tracker extras, the
 * Ultimate runtime, or the published LeRobot plugin while crossing into the
 * hardened installer layout.
 */
export const FIRST_SAFE_SELF_UPDATE_VERSION = "0.1.36"

function parseNumericVersion(version: string): number[] | null {
  const normalized = version.trim().replace(/^v/, "")
  if (!/^\d+(?:\.\d+)*$/.test(normalized)) return null
  const parts = normalized.split(".").map(Number)
  return parts.every(Number.isSafeInteger) ? parts : null
}

function compareNumericVersions(left: number[], right: number[]): number {
  for (let index = 0; index < Math.max(left.length, right.length); index++) {
    const difference = (left[index] ?? 0) - (right[index] ?? 0)
    if (difference !== 0) return Math.sign(difference)
  }
  return 0
}

/** Fail closed when the connected server must migrate through the installer. */
export function requiresInstallerMigration(version: string | null): boolean {
  if (version === null) return true
  const installed = parseNumericVersion(version)
  const firstSafe = parseNumericVersion(FIRST_SAFE_SELF_UPDATE_VERSION)
  return (
    installed === null || firstSafe === null || compareNumericVersions(installed, firstSafe) < 0
  )
}
