import type {
  CanDiscoveryState,
  CanProfileInventory,
  CanProfilePresence,
  HardwareProfile,
  RobotChannels,
  RobotState,
} from "./supervisor"

/** Prefer the backend's app lifetime; retain the failure heuristic for old hosts. */
export function canServerEpoch(
  serverInstanceId: string | null | undefined,
  legacyRecoveryEpoch: number
): string {
  return serverInstanceId
    ? `instance:${serverInstanceId}`
    : `legacy-recovery:${legacyRecoveryEpoch}`
}

/** Discovery must settle before any saved-profile auto-connect decision. */
export function canDiscoveryBlocksAutoConnect(
  discovery: CanDiscoveryState | null | undefined
): boolean {
  return (
    discovery?.status === "needed" ||
    discovery?.status === "running" ||
    discovery?.status === "unidentified" ||
    discovery?.status === "error"
  )
}

/**
 * Start discovery only from one coherent idle snapshot. A failed earlier
 * connect may leave RobotLink in `error`; the backend retries disconnect and
 * independently proves the link is fully released before touching setup.
 * It also repeats the ownership checks and single-flights requests from tabs.
 */
export function shouldStartCanDiscovery(
  discovery: CanDiscoveryState | null | undefined,
  robotState: RobotState | undefined,
  pollStateKnown: boolean,
  hardwareIdle: boolean
): boolean {
  return (
    discovery?.status === "needed" &&
    (robotState === "disconnected" || robotState === "error") &&
    pollStateKnown &&
    hardwareIdle
  )
}

/** One automatic request per server-owned unresolved-hardware generation. */
export function canDiscoveryAttemptSignature(
  discovery: CanDiscoveryState | null | undefined
): string | null {
  return discovery?.status === "needed" ? String(discovery.generation) : null
}

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

/**
 * Ordering guard for the 2 s CAN inventory poll. `issued` is the id of the
 * latest request sent; `applied` is the id of the latest response applied to
 * state (or a watermark below which every outstanding request is void).
 *
 * The poll is a fixed interval that does not wait for in-flight requests, and
 * `/api/can/interfaces` shells out to `udevadm` per netdev — slow exactly
 * during CAN discovery. Dropping every response that isn't the *newest issued*
 * request would starve inventory forever once latency exceeds the cadence, so
 * only responses older than the last *applied* one are discarded.
 */
export type CanInventoryPollSequence = { issued: number; applied: number }

export function newCanInventoryPollSequence(): CanInventoryPollSequence {
  return { issued: 0, applied: 0 }
}

/** Issue a new request id for one poll. */
export function issueCanInventoryPoll(sequence: CanInventoryPollSequence): number {
  sequence.issued += 1
  return sequence.issued
}

/**
 * Claim the right to apply `requestId`'s response. True (and records it as
 * applied) unless a newer response has already landed or the request was
 * voided by `voidCanInventoryPolls`; false means drop the response.
 */
export function claimCanInventoryPollResponse(
  sequence: CanInventoryPollSequence,
  requestId: number
): boolean {
  if (requestId <= sequence.applied) return false
  sequence.applied = requestId
  return true
}

/** Void every outstanding request, e.g. after a disconnect or host change. */
export function voidCanInventoryPolls(sequence: CanInventoryPollSequence): void {
  sequence.applied = sequence.issued
}

/** Auto-connect requires one coherent successful snapshot of all authorities. */
export function autoConnectPollStateKnown(
  robotStatusKnown: boolean,
  canInventoryKnown: boolean,
  sessionInventoryKnown: boolean
): boolean {
  return robotStatusKnown && canInventoryKnown && sessionInventoryKnown
}
