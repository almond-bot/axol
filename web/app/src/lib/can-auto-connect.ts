import type {
  CanProfileInventory,
  CanProfilePresence,
  HardwareProfile,
  RobotChannels,
} from "./supervisor"

/**
 * Pick the one idle telemetry profile that should be connected automatically.
 * The backend has already matched physical interfaces to the operator's saved
 * Axol and Mantis channel maps, so arbitrary SocketCAN names are never guessed.
 */
export function chooseAutoConnectProfile(
  profiles: CanProfileInventory,
  preferred: HardwareProfile
): HardwareProfile | null {
  let target: HardwareProfile | null = null
  if (profiles[preferred].present) target = preferred
  const other: HardwareProfile = preferred === "axol" ? "mantis" : "axol"
  if (target === null && profiles[other].present) target = other
  if (target === null || profiles[target].automaticConnectSuppressed) return null
  return target
}

/** Select hardware only when host inventory makes the role unambiguous. */
export function chooseUnambiguousAutoConnectProfile(
  profiles: CanProfileInventory
): HardwareProfile | null {
  const axol = profiles.axol.present
  const mantis = profiles.mantis.present
  if (axol === mantis) return null
  const target = axol ? "axol" : "mantis"
  return profiles[target].automaticConnectSuppressed ? null : target
}

/**
 * Resolve Diagnostics' deliberately narrower startup policy. New hosts must
 * prove that exactly one configured profile is present. A legacy host cannot
 * provide that inventory, so retain its historical server-reported profile;
 * when old status omits the field the caller supplies Axol, never a guessed
 * Mantis role. Unknown/failed inventory remains fail-closed.
 */
export function chooseDiagnosticsAutoConnectProfile(
  profiles: CanProfileInventory | null,
  legacyInventory: boolean,
  activeProfile: HardwareProfile
): HardwareProfile | null {
  if (profiles !== null) return chooseUnambiguousAutoConnectProfile(profiles)
  return legacyInventory ? activeProfile : null
}

/**
 * Choose a detected profile, except that an open link on an obsolete saved map
 * must be moved off that map even when its replacement interfaces are absent.
 * The resulting connect closes the stale buses and leaves an explicit error on
 * the requested map; ordinary hardware absence remains passive.
 */
export function chooseAutoConnectTarget(
  profiles: CanProfileInventory,
  preferred: HardwareProfile,
  activeProfile: HardwareProfile,
  activeChannels: RobotChannels | undefined,
  activeConnected: boolean
): HardwareProfile | null {
  const detected = chooseAutoConnectProfile(profiles, preferred)
  if (detected !== null) return detected
  if (!activeConnected || activeChannels === undefined) return null
  const configured = profiles[activeProfile].channels
  return activeChannels.left !== configured.left || activeChannels.right !== configured.right
    ? activeProfile
    : null
}

/** A mapping or administrative link-state change permits a fresh retry set. */
export function autoConnectSignature(
  profile: HardwareProfile,
  presence: CanProfilePresence
): string {
  return JSON.stringify([profile, presence.channels.left, presence.channels.right, presence.up])
}

/** Initial attempt is immediate; the next two are delayed, then retries stop. */
export function autoConnectRetryDelay(attempt: number): number | null {
  return attempt === 1 ? 2000 : attempt === 2 ? 5000 : null
}

/** Admit at most three calls for one profile/mapping/link-state signature. */
export function nextAutoConnectAttempt(
  previousAttempts: number,
  pollStateKnown = true
): number | null {
  if (!pollStateKnown) return null
  return previousAttempts < 3 ? previousAttempts + 1 : null
}

/** Auto-connect requires one coherent successful snapshot of all authorities. */
export function autoConnectPollStateKnown(
  robotStatusKnown: boolean,
  canInventoryKnown: boolean,
  sessionInventoryKnown: boolean
): boolean {
  return robotStatusKnown && canInventoryKnown && sessionInventoryKnown
}
