import {
  BrainCircuit,
  CircleDot,
  Gamepad2,
  Hand,
  History,
  Merge,
  Move3d,
  Play,
  Route,
  Smartphone,
  type LucideIcon,
} from "lucide-react"
import type { OperationId, OperationMeta } from "@/lib/supervisor"

/**
 * The operations as the app presents them: names, icons, groups, one-line
 * summaries and what each needs. The app owns this, so the Operations screen
 * looks the same before the Host Computer is connected; the host only adds
 * what it knows (which operations it can run, their per-run fields, their
 * status). An operation a newer host adds that isn't listed here still shows,
 * under "More", with the host's own wording.
 */
export interface CatalogEntry {
  id: OperationId
  label: string
  /** One sentence: what the operation does. */
  summary: string
  icon: LucideIcon
  group: OperationGroup
  /** What the operator brings, shown on the tile ("VR headset", "Cameras"). */
  needs: string[]
}

export type OperationGroup = "drive" | "record" | "playback" | "more"

export const OPERATION_GROUPS: { key: OperationGroup; label: string }[] = [
  { key: "drive", label: "Drive" },
  { key: "record", label: "Record" },
  { key: "playback", label: "Play back" },
  { key: "more", label: "More" },
]

export const OPERATION_CATALOG: CatalogEntry[] = [
  {
    id: "teleop",
    label: "Teleoperation",
    summary: "Drive Axol from a VR headset.",
    icon: Gamepad2,
    group: "drive",
    needs: ["VR headset"],
  },
  {
    id: "gravity-comp",
    label: "Gravity compensation",
    summary: "Make the arms weightless so you can move them by hand.",
    icon: Hand,
    group: "drive",
    needs: [],
  },
  {
    id: "waypoints",
    label: "Waypoints",
    summary: "Record poses by hand, then replay them as a path.",
    icon: Route,
    group: "drive",
    needs: [],
  },
  {
    id: "jog",
    label: "Jog",
    summary: "Drag a gripper to a pose in 3D, and drive the lift and Jelly.",
    icon: Move3d,
    group: "drive",
    needs: [],
  },
  {
    id: "marionette",
    label: "Marionette",
    summary: "Drive the arms, lift and Jelly from iPhones running Marionette.",
    icon: Smartphone,
    group: "drive",
    needs: ["iPhone with Marionette"],
  },
  {
    id: "collect-data",
    label: "Collect data",
    summary: "Record teleoperated episodes into a LeRobot dataset.",
    icon: CircleDot,
    group: "record",
    needs: ["VR headset", "Cameras"],
  },
  {
    id: "collect-dagger",
    label: "Collect DAgger data",
    summary: "Run a policy and take over with the controllers to correct it.",
    icon: Merge,
    group: "record",
    needs: ["Trained policy", "VR headset", "Cameras"],
  },
  {
    id: "replay-dataset",
    label: "Replay dataset",
    summary: "Play a recorded episode back on the arms.",
    icon: History,
    group: "playback",
    needs: ["Dataset"],
  },
  {
    id: "run-policy",
    label: "Run policy",
    summary: "Run a trained policy on Axol and record what it does.",
    icon: BrainCircuit,
    group: "playback",
    needs: ["Trained policy", "Cameras"],
  },
]

const BY_ID = new Map(OPERATION_CATALOG.map((e) => [e.id, e]))

/** The catalog entry for an operation, or a generic one built from what the
 *  host says about an operation the app doesn't know yet. */
export function catalogEntry(id: OperationId, hostMeta?: OperationMeta | null): CatalogEntry {
  return (
    BY_ID.get(id) ?? {
      id,
      label: hostMeta?.label ?? id,
      summary: hostMeta?.description ?? "",
      icon: Play,
      group: "more",
      needs: [],
    }
  )
}

/**
 * An operation the host doesn't report, described well enough to open its
 * screen: before a host is connected, or on a host without it. Start stays
 * off, since there is no host spec to run.
 */
export function catalogMeta(id: OperationId): OperationMeta | null {
  const entry = BY_ID.get(id)
  if (!entry) return null
  return {
    id,
    label: entry.label,
    description: entry.summary,
    fields: [],
    suggestedFields: [],
    strictFields: [],
    requiresRobot: true,
    requiresCameras: entry.needs.includes("Cameras"),
    simCapable: false,
    simFlag: null,
    robotFreeFlags: [],
    armsFlag: null,
    supportsMantis: false,
    episodeControl: false,
    usesHeadset: entry.needs.includes("VR headset"),
    streamsVideo: false,
  }
}

/**
 * The operations to show, in catalog order, then any the host adds. Before a
 * host is connected every catalog operation shows. Once connected, the host's
 * list decides for Mantis (which runs only some operations); for Axol, catalog
 * operations the host lacks still show, marked as not on this robot.
 */
export function launcherEntries({
  hostOperations,
  connected,
  mantis,
}: {
  hostOperations: OperationMeta[]
  connected: boolean
  mantis: boolean
}): CatalogEntry[] {
  const hostIds = new Set(hostOperations.map((o) => o.id))
  const known = OPERATION_CATALOG.filter((e) => !connected || !mantis || hostIds.has(e.id))
  const extra = hostOperations.filter((o) => !BY_ID.has(o.id)).map((o) => catalogEntry(o.id, o))
  return [...known, ...extra]
}

export type OperationStatusTone = "running" | "ready" | "attention" | "off"

export interface OperationStatus {
  tone: OperationStatusTone
  label: string
}

/**
 * The one-line status on an operation's tile. It covers what the app can
 * tell without opening the operation; the operation's screen lists every
 * check before Start.
 */
export function operationStatus({
  entry,
  hostMeta,
  available,
  connected,
  running,
  otherRunning,
  needsMount,
  hasCameras,
}: {
  entry: CatalogEntry
  /** The host's description of this operation, or null if it has none. */
  hostMeta: OperationMeta | null
  /** Whether the host can run it (its software and hardware are present). */
  available: boolean
  connected: boolean
  running: boolean
  otherRunning: boolean
  needsMount: boolean
  hasCameras: boolean
}): OperationStatus {
  if (running) return { tone: "running", label: "Running" }
  if (!connected) return { tone: "off", label: "Connect to start" }
  if (!hostMeta) return { tone: "off", label: "Not on this robot" }
  if (!available) return { tone: "off", label: "Not available" }
  if (otherRunning) return { tone: "off", label: "Another operation is running" }
  if (needsMount && hostMeta.requiresRobot) return { tone: "attention", label: "Choose a mount" }
  if (hostMeta.requiresCameras && !hasCameras) {
    return { tone: "attention", label: "Assign cameras" }
  }
  return { tone: "ready", label: entry.needs.length ? entry.needs.join(" · ") : "Ready" }
}

const RECENT_KEY = "axolRecentOperations"

/** Operations this browser opened most recently, newest first. */
export function recentOperations(): OperationId[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(RECENT_KEY) ?? "[]")
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === "string") : []
  } catch {
    return []
  }
}

export function rememberOperation(id: OperationId) {
  try {
    const next = [id, ...recentOperations().filter((o) => o !== id)].slice(0, 6)
    localStorage.setItem(RECENT_KEY, JSON.stringify(next))
  } catch {
    // Private mode or blocked storage: quick start falls back to defaults.
  }
}

/** The operations quick start offers: recent ones first, then the usual ones. */
export function quickStartOperations(
  entries: CatalogEntry[],
  recent: OperationId[],
  count = 3
): CatalogEntry[] {
  const order = [...recent, "teleop", "collect-data", "run-policy", "gravity-comp"]
  const picked: CatalogEntry[] = []
  for (const id of order) {
    const entry = entries.find((e) => e.id === id)
    if (entry && !picked.includes(entry)) picked.push(entry)
    if (picked.length === count) break
  }
  return picked
}
