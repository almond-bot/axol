import { useEffect, useRef, useState } from "react"

export type FieldType = "boolean" | "number" | "select" | "text" | "vector"

/** A single configurable leaf in a command's config (serve/introspect.py). */
export interface SchemaField {
  kind: "field"
  key: string
  label: string
  type: FieldType
  /** For "vector" fields the default is the numeric array itself. */
  default: string | number | boolean | number[] | null
  options?: string[] | null
  required: boolean
  /** Optional one-line help (argparse commands carry their flag help). */
  help?: string | null
}

/** A nested config section (a dataclass / dict in the config tree). */
export interface SchemaGroup {
  kind: "group"
  key: string
  label: string
  children: SchemaNode[]
}

export type SchemaNode = SchemaField | SchemaGroup

/** A launchable CLI command plus its full introspected config schema. */
export interface CommandSpec {
  id: string
  cli: string
  label: string
  description: string
  /** Catalog group: "Operate" | "Cameras" | "Calibrate" | "Setup". */
  category: string
  simCapable: boolean
  requiresHardware: boolean
  available: boolean
  error: string | null
  schema: SchemaNode[]
  required: string[]
  // The operation surface (serve/commands.py CommandDef). Optional because a
  // serve host older than the command registry doesn't send them — the panel
  // then falls back to its built-in operation list (see OPERATIONS).
  /** Runs in-process, so it gets an operation panel of its own. */
  isOperation?: boolean
  /** Needs at least one camera serial configured before it can start. */
  requiresCameras?: boolean
  /** Config keys the panel asks for per run; the rest come from Settings. */
  perRunFields?: string[]
  /** Drives episodes the panel can start / save / discard. */
  episodeControl?: boolean
  /** Arg name that means "no hardware", or null when the robot is required. */
  simFlag?: string | null
  /** Arg names that skip the arm-robot gates without being sim (cart_only). */
  robotFreeFlags?: string[]
  /** Whether this operation can run against the Mantis hardware profile. */
  supportsMantis?: boolean
  /** Connected hardware profiles on which this command may be launched. */
  hardwareProfiles?: HardwareProfile[]
  /** Driven from the VR headset, so the panel shows the connect hint. */
  usesHeadset?: boolean
  /** Honors the camera spec's headset-stream branch during this operation. */
  streamsVideo?: boolean
}

/** Catalog category display order (matches serve/commands.py CATEGORY_ORDER). */
export const CATEGORY_ORDER = ["Operate", "Diagnostics", "Calibrate", "Setup"]

export type SessionStatus = "starting" | "running" | "stopping" | "exited" | "error"

export interface SessionInfo {
  id: string
  command: string
  args: Record<string, unknown>
  status: SessionStatus
  exitCode: number | null
  error: string | null
  startedAt: number
  pid: number | null
}

/** A submitted form value; vector fields carry one entry per component
 * (numbers once parseable, the raw text while mid-edit). */
export type FormValue = string | boolean | (number | string)[]

const MAX_LINES = 5000

// All API/WebSocket calls target this base (the machine running `axol serve`).
// Empty means same-origin — used when the panel is served by that machine
// directly; the hosted site (axol.almond.bot) sets it to the entered address.
let apiBase = ""

/** Point the client at a serve address (host, host:port, or full URL). */
export function setServerBase(host: string): void {
  apiBase = serverHttpBase(host)
}

/**
 * Normalize a user-entered address to an `https://host:port` origin (or "").
 * Defaults to HTTPS + port 8001 since the local serve is TLS by default and an
 * HTTPS page cannot call a plain-HTTP server (mixed content).
 */
export function serverHttpBase(host: string): string {
  const h = host.trim()
  if (!h) return ""
  const withScheme = /^https?:\/\//.test(h) ? h : `https://${h}`
  try {
    const u = new URL(withScheme)
    if (!u.port) u.port = "8001"
    return u.origin
  } catch {
    return ""
  }
}

export function apiUrl(path: string): string {
  return `${apiBase}${path}`
}

/**
 * Whether this bundle is the one `axol serve` hosts from web/app/dist — a
 * commit mismatch with the backend then means that local bundle is stale.
 * Only a production build on the backend's own origin qualifies: the Vite dev
 * server also proxies /api same-origin, but it serves its own working-tree
 * code, not the backend's dist, so dev is never "served by the backend".
 * An explicit server base still counts when it points back at the origin
 * that served the page (e.g. a host entered on this panel earlier).
 */
export function servedByBackend(): boolean {
  return !import.meta.env.DEV && (apiBase === "" || apiBase === window.location.origin)
}

/** WebSocket origin for the current server base (ws(s)://host[:port]). */
export function wsBaseUrl(): string {
  const base = apiBase || window.location.origin
  const u = new URL(base)
  const proto = u.protocol === "https:" ? "wss" : "ws"
  return `${proto}://${u.host}`
}

export class ApiRequestError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = "ApiRequestError"
    this.status = status
  }
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiRequestError(
      (body as { error?: string }).error ?? `HTTP ${res.status}`,
      res.status
    )
  }
  // A non-JSON 200 means whatever answered isn't axol serve (typically the
  // static site itself when no host is configured, answering index.html for
  // /api/*) — say that instead of surfacing a JSON SyntaxError.
  try {
    return (await res.json()) as T
  } catch {
    throw new Error("the host did not answer like an axol serve host — set the axol host address")
  }
}

export interface ServerInfo {
  hostname: string
  lanIp: string
  viewerPort: number
  vrPort: number
  /** Installed release version of the serve host, e.g. "0.1.2". */
  version?: string | null
  /** Git commit the serve host is running (PEP 610 pin or checkout HEAD). */
  commit?: string | null
  /** Tag-pinned git tool install (true) vs dev checkout (false). */
  releaseInstall?: boolean
}

export async function fetchInfo(): Promise<ServerInfo> {
  return json(await fetch(apiUrl("/api/info")))
}

// ---------------------------------------------------------------------------
// Self-update (read-only release-tag check + user-initiated upgrade)
// ---------------------------------------------------------------------------

export type UpdateState = "idle" | "updating" | "error"

/** Current step while an update is applying; null when not updating. */
export type UpdatePhase = "upgrading" | "provisioning" | "restarting"

export interface UpdateStatus {
  /** Updatable: installed from git as a uv tool with uv available. */
  enabled: boolean
  /** Installed release version, e.g. "0.1.2". */
  version: string | null
  /** Latest release version (highest vX.Y.Z tag), or null until first resolved/offline. */
  remoteVersion: string | null
  /** A release with a higher version than the installed one exists. */
  updateAvailable: boolean
  /** Safe to restart now (no op running). */
  idle: boolean
  state: UpdateState
  /** Step while state is "updating" (upgrading/provisioning/restarting); else null. */
  phase: UpdatePhase | null
  /** Last update failure, surfaced to the operator; null otherwise. */
  error: string | null
}

/**
 * Fetch the update indicator. Pass `force` on connect / page load to make the
 * server resolve the latest release tag synchronously (bypassing its debounce),
 * so the result is immediately current rather than a stale cached value.
 */
export async function fetchUpdateStatus(force = false): Promise<UpdateStatus> {
  return json(await fetch(apiUrl(`/api/update/status${force ? "?refresh=1" : ""}`)))
}

/**
 * Probe the update API before touching legacy polling endpoints. Servers from
 * v0.1.0 through v0.1.2 return 404 here, but `/api/info` and `/api/op/status`
 * trigger their unsafe main-tracking updater. `null` therefore means the UI
 * must stop and show the one-time hosted-installer migration.
 */
export async function probeUpdateStatus(): Promise<UpdateStatus | null> {
  let response: Response
  try {
    response = await fetch(apiUrl("/api/update/status"), {
      signal: AbortSignal.timeout(CONNECT_TIMEOUT_MS),
    })
  } catch (error) {
    if (
      error instanceof DOMException &&
      (error.name === "TimeoutError" || error.name === "AbortError")
    ) {
      throw new Error(`no response from the host after ${CONNECT_TIMEOUT_MS / 1000}s`)
    }
    throw error
  }
  if (response.status === 404) return null
  return json(response)
}

/** Trigger the on-demand upgrade; the server restarts onto new code when idle. */
export async function startUpdate(): Promise<{ started: boolean }> {
  return json(await fetch(apiUrl("/api/update/start"), { method: "POST" }))
}

/**
 * Power off the serve host (`shutdown -h now`). Refused (409) while an
 * operation or session is running.
 */
export async function shutdownHost(): Promise<{ ok: boolean }> {
  return json(await fetch(apiUrl("/api/host/shutdown"), { method: "POST" }))
}

/**
 * Reboot the serve host (`shutdown -r now`). Refused (409) while an
 * operation or session is running.
 */
export async function restartHost(): Promise<{ ok: boolean }> {
  return json(await fetch(apiUrl("/api/host/restart"), { method: "POST" }))
}

// ---------------------------------------------------------------------------
// Robot connection (detached CAN + 1 Hz motor ping)
// ---------------------------------------------------------------------------

export type RobotState = "disconnected" | "connecting" | "connected" | "busy" | "error"
export type HardwareProfile = "axol" | "mantis"

/**
 * The system-wide device selection — which hardware every operation runs on.
 * Stored with the shared settings on the serve host under this key (so every
 * operator device agrees), mirrored to localStorage for hosts too old to have
 * it, and translated into each run's `mantis` flag at start.
 */
export const HARDWARE_PROFILE_SETTING = "system.hardware_profile"
/** The per-run config flag the device selection becomes on Mantis-capable ops. */
export const HARDWARE_PROFILE_ARG = "mantis"
const HARDWARE_PROFILE_STORAGE = "axolHardwareProfile"

export function parseHardwareProfile(value: unknown): HardwareProfile | null {
  return value === "axol" || value === "mantis" ? value : null
}

export function loadLocalHardwareProfile(): HardwareProfile {
  try {
    return parseHardwareProfile(localStorage.getItem(HARDWARE_PROFILE_STORAGE)) ?? "axol"
  } catch {
    return "axol"
  }
}

export function saveLocalHardwareProfile(profile: HardwareProfile): void {
  try {
    localStorage.setItem(HARDWARE_PROFILE_STORAGE, profile)
  } catch {
    // ignore storage failures (private mode / quota)
  }
}

/** Per-run flags that only make sense on the Axol profile (sim / cart-only
 *  drive the arm simulator or the cart, never the handheld rigs). */
const AXOL_ONLY_RUN_FLAGS = new Set(["sim", "cart_only"])

/**
 * Whether a per-run field is shown/sent for the given device. The legacy
 * `mantis` toggle is never per-run any more, and Axol-only run modes are
 * hidden while Mantis is the selected device.
 */
export function runFieldVisible(key: string, profile: HardwareProfile): boolean {
  if (key === HARDWARE_PROFILE_ARG) return false
  if (profile === "mantis" && AXOL_ONLY_RUN_FLAGS.has(key)) return false
  return true
}

export interface MotorHealth {
  arm: string
  joint: string
  reachable: boolean
  /** MotorStatus name from the idle ping (e.g. "OK", "OVER_TEMPERATURE"). */
  status: string | null
  temperature: number | null
  voltage: number | null
}

/** A motor an operation must not drive through: unreachable or errored. */
export interface MotorFault {
  arm: string
  joint: string
  /** Human-readable problem, e.g. "unreachable", "over temperature". */
  problem: string
  temperature: number | null
}

/** The CAN interfaces the robot link opens; null = that arm is disabled. */
export interface RobotChannels {
  left: string | null
  right: string | null
}

export interface RobotStatus {
  state: RobotState
  connected: boolean
  error: string | null
  lastPing: number | null
  motors: MotorHealth[]
  motorCount: number
  reachableCount: number
  /** Faulted motors while connected (server-computed); [] otherwise. */
  faults?: MotorFault[]
  /** Configured CAN interfaces (older hosts omit this). */
  channels?: RobotChannels
  /** Hardware currently shown by the idle diagnostics link. */
  profile?: HardwareProfile
  /** Whether this robot has grippers (older hosts omit this = true). */
  hasGripper?: boolean
}

/** Short display label for a fault, e.g. "L elbow — over temperature (78°C)". */
export function motorFaultLabel(f: MotorFault): string {
  const joint = f.joint.replace(/_/g, " ").toLowerCase()
  const temp =
    f.problem.includes("temp") && f.temperature != null ? ` (${Math.round(f.temperature)}°C)` : ""
  return `${f.arm[0].toUpperCase()} ${joint} — ${f.problem}${temp}`
}

export async function fetchRobotStatus(): Promise<RobotStatus> {
  return json(await fetch(apiUrl("/api/robot/status")))
}

/**
 * Connect the robot link. An explicit `channels` selection (the CAN adapter
 * picker, used when the Axol hub adapter isn't present) is persisted on the
 * host and reused by every later connect and operation; omit it to connect
 * with the stored/default interfaces.
 */
export async function robotConnect(
  channels?: RobotChannels,
  profile: HardwareProfile = "axol",
  automatic = false
): Promise<RobotStatus> {
  const init: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      leftChannel: channels?.left ?? null,
      rightChannel: channels?.right ?? null,
      channelsSet: channels != null,
      profile,
      automatic,
    }),
  }
  return json(await fetch(apiUrl("/api/robot/connect"), init))
}

export async function robotDisconnect(): Promise<RobotStatus> {
  return json(await fetch(apiUrl("/api/robot/disconnect"), { method: "POST" }))
}

/** One SocketCAN interface on the serve host. */
export interface CanInterface {
  name: string
  up: boolean
}

/** Presence of one configured hardware profile in the host's CAN inventory. */
export interface CanProfilePresence {
  channels: RobotChannels
  /** Configured channels exist, or their exact persisted USB hub is attached. */
  present: boolean
  /** Every configured channel netdev exists and is administratively up. */
  up: boolean
  /** This exact profile/map was manually disconnected on the serve host. */
  automaticConnectSuppressed?: boolean
}

export type CanProfileInventory = Record<HardwareProfile, CanProfilePresence>

export type CanDiscoveryStatus =
  | "ready"
  | "needed"
  | "running"
  | "configured"
  | "partial"
  | "unidentified"
  | "error"

/** Server-owned discovery of an attached, not-yet-trusted Axol/Mantis hub. */
export interface CanDiscoveryState {
  status: CanDiscoveryStatus
  candidateCount: number
  /** Opaque server-local epoch; changes when unresolved attached hardware changes. */
  generation: number
  message?: string
}

export interface CanInterfaceInventory {
  /** Opaque app-lifetime identity; omitted by older serve releases. */
  serverInstanceId?: string
  interfaces: CanInterface[]
  /** Omitted by serve releases that predate hardware-aware auto-connect. */
  profiles?: CanProfileInventory
  /** Omitted by serve releases that predate safe, non-interactive discovery. */
  discovery?: CanDiscoveryState
}

export async function fetchCanInterfaces(): Promise<CanInterfaceInventory> {
  return json(await fetch(apiUrl("/api/can/interfaces")))
}

/** Identify and persist any attached, unassigned Axol/Mantis hub. */
export async function discoverCanHardware(force = false): Promise<CanInterfaceInventory> {
  const path = force ? "/api/can/discover?force=true" : "/api/can/discover"
  return json(await fetch(apiUrl(path), { method: "POST" }))
}

/** A busy race, transport interruption, or server fault may retry on the next inventory poll. */
export function canDiscoveryRequestCanRetry(error: unknown): boolean {
  return (
    error instanceof TypeError ||
    (error instanceof ApiRequestError && (error.status === 409 || error.status >= 500))
  )
}

// ---------------------------------------------------------------------------
// Local ZED cameras (attached to the serve machine)
// ---------------------------------------------------------------------------

/** One ZED camera detected on the serve machine. */
export interface CameraDevice {
  serial: number
  model: string
  /** "mono" (ZED-X One) | "stereo" (ZED X). */
  kind: string
}

export interface CameraDetectResult {
  devices: CameraDevice[]
  /** Why detection failed (e.g. pyzed not installed); null when it worked. */
  error: string | null
}

export async function detectCameras(): Promise<CameraDetectResult> {
  return json(await fetch(apiUrl("/api/cameras/detect")))
}

export async function restartCameraDaemon(): Promise<{ ok: boolean; error: string | null }> {
  return json(await fetch(apiUrl("/api/cameras/restart-daemon"), { method: "POST" }))
}

// -- Quest over USB (adb reverse pose tunnel) -----------------------------

export interface UsbStatus {
  installed: boolean
  serial: string | null
  // "none" | "device" | "unauthorized" | "offline" | raw adb state string
  state: string
  reverseActive: boolean
  ready: boolean
}

export async function fetchUsbStatus(): Promise<UsbStatus> {
  return json(await fetch(apiUrl("/api/usb/status")))
}

export async function usbConnect(): Promise<UsbStatus> {
  return json(await fetch(apiUrl("/api/usb/connect"), { method: "POST" }))
}

/**
 * Disable (true) or restore (false) the headset's proximity sensor over adb.
 * Disabled, the Quest stays awake with nobody wearing it — headless sessions
 * keep their pose stream. Holds until restored or the headset reboots.
 */
export async function setQuestProximityDisabled(disabled: boolean): Promise<{ ok: boolean }> {
  return json(
    await fetch(apiUrl("/api/usb/proximity"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ disabled }),
    })
  )
}

// ---------------------------------------------------------------------------
// In-process operations (teleop / gravity-comp / collect-data / run-policy)
// ---------------------------------------------------------------------------

/** An in-process operation's id. Open-ended: the set comes from the connected
 *  serve host's command registry, which a downstream package can add to. */
export type OperationId = string

/** Episode lifecycle phase, surfaced so the control panel on any
 *  computer shows the right episode controls (not just the tab that started it).
 *  Open-ended: an episode control can define its own phases (collect-data's
 *  "countdown" / "contact", or a downstream package's) — the panel renders the
 *  server-driven `message`/`controls` then and never needs to know them. */
export type PolicyPhase = string

/** One episode-control button the running op offers right now (server-driven). */
export interface EpisodeControlSpec {
  /** Command pushed to /api/op/episode when clicked. */
  command: string
  label: string
  /** Arm on first click and require a second, confirming click — the panel's
   *  stand-in for the headset's double-press save/discard confirmation. */
  confirm?: boolean
  /** Render as a text field + submit button instead of a plain button;
   *  submitting posts `${command} ${text}` (e.g. a downstream op's
   *  post-episode notes). */
  input?: boolean
  /** The input's placeholder — doubles as the operator instruction (e.g.
   *  which episode the text attaches to). */
  placeholder?: string
  /** The current server-side text, prefilled whenever the input (re)appears
   *  so a submitted value survives phase changes and can be edited. */
  value?: string
}

export interface PolicyState {
  phase: PolicyPhase
  episodesRecorded: number
  /** Operator status line; when present it replaces the panel's built-in
   *  phase text (lets the op mirror what the headset HUD would say). */
  message?: string
  /** Buttons to render; when present they replace the panel's built-in
   *  run-policy Start/Save/Discard set. */
  controls?: EpisodeControlSpec[]
  /** 1-based number of the episode being recorded next/now, when the op
   *  tracks a dataset episode index (mirrors the headset HUD readout). */
  episode?: number
}

export interface OpStatus {
  running: boolean
  session: SessionInfo | null
  /** Present only while an op declaring an episode control is running
   *  (collect-data / run-policy / waypoints); null otherwise. */
  policy: PolicyState | null
}

export async function fetchOpStatus(): Promise<OpStatus> {
  return json(await fetch(apiUrl("/api/op/status")))
}

export async function startOperation(
  op: OperationId,
  args: Record<string, FormValue>,
  cameras?: CameraSpec
): Promise<SessionInfo> {
  return json(
    await fetch(apiUrl("/api/op/start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ op, args, cameras: cameras ?? null }),
    })
  )
}

export async function stopOperation(): Promise<SessionInfo> {
  return json(await fetch(apiUrl("/api/op/stop"), { method: "POST" }))
}

/** run-policy episode control: ``start`` | ``s`` (save) | ``r`` (rerecord) | ``q`` (quit). */
export async function sendEpisodeCommand(command: string): Promise<{ ok: boolean }> {
  return json(
    await fetch(apiUrl("/api/op/episode"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command }),
    })
  )
}

// ---------------------------------------------------------------------------
// Datasets on disk (the operation panels' shared repo-id picker)
// ---------------------------------------------------------------------------

/** One LeRobot dataset found on the serve host (see /api/datasets). */
export interface DatasetInfo {
  /** Repo id relative to the datasets root — accepted by dataset operations. */
  repoId: string
  /** Absolute dataset directory on the serve host. */
  root: string
  episodes: number | null
  fps: number | null
}

/** Datasets on the serve host, newest first. Empty on older hosts (404). */
export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const res: { datasets?: DatasetInfo[] } = await json(await fetch(apiUrl("/api/datasets")))
  return res.datasets ?? []
}

/** Which eye(s) of a stereo ZED X to use, per branch. */
export type StereoEyes = "both" | "left" | "right"

export type CameraSlot = "overhead" | "left_arm" | "right_arm"

/** Per-camera participation in a branch (streaming or recording):
 * - `false` — camera opted out of this branch.
 * - `true` — mono camera opted in.
 * - an eye name — stereo camera opted in with that eye selection.
 */
export type BranchSel = boolean | StereoEyes

/** "off" disables a whole branch (streaming or recording) globally. */
export const RESOLUTION_OFF = "off"

/** Local camera setup forwarded with op starts (see serve/runner.py).
 *
 * The serials map camera slots to the ZED cameras attached to the serve
 * machine. Streaming (the headset feed) and recording (the dataset) are
 * configured independently — globally and per camera:
 *
 * - `stream_resolution` / `record_resolution` set the capture and dataset
 *   resolutions; `"off"` disables that whole branch.
 * - `stream` / `record` decide, per slot, whether the camera takes part (and,
 *   for stereo, which eyes) — so an operator can stream both eyes for depth
 *   while recording one, stream a camera without recording it (or vice versa),
 *   or turn either branch off entirely.
 */
export interface CameraSpec {
  serials: Record<CameraSlot, string>
  /** Mantis wrist-camera assignment, kept separate from the Axol camera map. */
  mantis_serials?: Pick<Record<CameraSlot, string>, "left_arm" | "right_arm">
  /** Capture resolution → headset stream (full quality), or `"off"`. */
  stream_resolution?: string
  /** Dataset downscale target (collect-data recording), or `"off"`. */
  record_resolution?: string
  /** Per-slot streaming participation (headset feed). */
  stream?: Partial<Record<CameraSlot, BranchSel>>
  /** Per-slot recording participation (dataset). */
  record?: Partial<Record<CameraSlot, BranchSel>>
  /** @deprecated legacy single resolution; read as `stream_resolution`. */
  resolution?: string
}

/**
 * How long the reachability probe waits before declaring the host absent.
 * Without this, a fetch to an absent host sits in TCP retries for a minute or
 * more with the UI stuck on "Connecting…". Kept just high enough for a live
 * LAN host's worst case (mDNS .local resolution + TCP + TLS handshake over
 * Wi-Fi); much lower and slow-but-alive hosts get misreported as offline.
 */
const CONNECT_TIMEOUT_MS = 2_000

export async function fetchCommands(): Promise<CommandSpec[]> {
  // Fail fast on an absent host by racing the timer against a probe that
  // answers from memory (/api/op/status — any HTTP response, even a 404 from
  // an older host, proves it's alive). The timer must not cover /api/commands
  // itself: its first call after a serve start imports every command's config
  // module to build the schemas and can take well past the timeout, so racing
  // it misreported a slow-but-alive host as offline — the first connect after
  // a cert authorization always failed and only the retry (against the
  // then-warm schema cache) got through.
  try {
    await fetch(apiUrl("/api/op/status"), {
      signal: AbortSignal.timeout(CONNECT_TIMEOUT_MS),
    })
  } catch (e) {
    if (e instanceof DOMException && (e.name === "TimeoutError" || e.name === "AbortError")) {
      throw new Error(`no response from the host after ${CONNECT_TIMEOUT_MS / 1000}s`)
    }
    throw e
  }
  return json(await fetch(apiUrl("/api/commands")))
}

export async function fetchSessions(): Promise<SessionInfo[]> {
  return json(await fetch(apiUrl("/api/sessions")))
}

export async function runCommand(
  command: string,
  args: Record<string, FormValue>
): Promise<SessionInfo> {
  return json(
    await fetch(apiUrl("/api/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, args }),
    })
  )
}

export async function stopSession(id: string): Promise<SessionInfo> {
  return json(await fetch(apiUrl(`/api/sessions/${id}/stop`), { method: "POST" }))
}

/** Answer a session's interactive prompt (empty line = a bare "Enter"). */
export async function sendSessionInput(id: string, line = ""): Promise<{ ok: boolean }> {
  return json(
    await fetch(apiUrl(`/api/sessions/${id}/input`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ line }),
    })
  )
}

export type TrackerBackend = "survive" | "ultimate"

export interface TrackerBinding {
  complete: boolean
  left: string | null
  right: string | null
}

export type TrackerTransformStatus = "measured" | "factory" | "candidate" | "missing" | "stale"

export interface TrackerTransformReadiness {
  left: TrackerTransformStatus
  right: TrackerTransformStatus
}

export interface QuestTrackerReadiness {
  binding: "automatic-handedness"
  installed: boolean
  transforms: TrackerTransformReadiness
  calibrationKey: string | null
  controllerProfile: string | null
  poseSpace: "grip" | "target-ray" | null
  availableCalibrationKeys: string[]
  datumStatus: "configured" | "ambiguous" | "invalid" | "missing"
  liveDatum?: {
    left: { profile: string | null; poseSpace: "grip" | "target-ray" | null }
    right: { profile: string | null; poseSpace: "grip" | "target-ray" | null }
    commonKey: string | null
    observedAt: number
    ageSeconds: number
    live: boolean
  } | null
}

export interface LighthouseTrackerReadiness {
  binding: TrackerBinding
  installed: boolean
  available: boolean
  pairingCli: boolean
  pinnedBuild: boolean
  udevReady: boolean
  pinnedRef: string
  buildRevision: string
  installedRef: string | null
  installedBuildRevision: string | null
  issues: string[]
  transforms: TrackerTransformReadiness
  /**
   * Last base-station survey from Check base stations or Identify trackers.
   * `null` until one has run on this host; absent on older serve hosts.
   */
  baseStations?: LighthouseBaseStationSurvey | null
}

export interface LighthouseBaseStationSurvey {
  checkedAt: number
  /**
   * Serials of the base stations a tracker actually received during the check,
   * keyed by the channel number shown on the station (1–16).
   */
  channels: Record<string, string[]>
  /** Stations libsurvive replayed from its saved calibration; not counted as seen. */
  savedChannels?: Record<string, string[]>
  clashingChannels: number[]
  baseStationCount: number
  /** How many stations the rig is expected to have; absent in older surveys. */
  expectedBaseStations?: number
  trackers: string[]
  /** Operator-facing problems, each ending with the fix. */
  problems: string[]
}

export interface UltimateTrackerReadiness {
  binding: TrackerBinding
  installed: boolean
  nativeDependencies: boolean
  pythonHid: boolean
  apiCompatible: boolean
  pinnedPyvut: boolean
  pinnedRef: string
  logSuppression: boolean
  udevReady: boolean
  operatorAccess: boolean
  dongleConnected: boolean
  endpointStatus: "accessible" | "permission-denied" | "missing" | "unavailable"
  wifiConfig: "valid" | "missing" | "invalid" | "permissions-warning"
  quatOrder: "xyzw" | "wxyz"
  upAxis: "y" | "z"
  issues: string[]
  transforms: TrackerTransformReadiness
}

export interface TrackerSourceReadiness {
  quest: QuestTrackerReadiness
  lighthouse: LighthouseTrackerReadiness
  ultimate: UltimateTrackerReadiness
}

export interface TrackerBindingsSnapshot {
  bindings: Record<TrackerBackend, TrackerBinding>
  /** Non-invasive setup checks; absent on older serve hosts. */
  sources?: TrackerSourceReadiness
  /** Resolved trigger/gripper channels; absent on older serve hosts. */
  channels?: RobotChannels
}

export async function fetchTrackerBindings(): Promise<TrackerBindingsSnapshot> {
  return json(await fetch(apiUrl("/api/tracker/bindings"), { cache: "no-store" }))
}

export type MantisTrackerSource = "quest" | "lighthouse" | "ultimate"

export interface UltimateWifiConfig {
  /** Host path is informational only; credential values never include the password. */
  path: string
  configured: boolean
  status: "valid" | "missing" | "invalid" | "permissions-warning"
  error: string | null
  ssid: string
  country: string
  freq: number
  passwordSet: boolean
}

export interface UltimateWifiUpdate {
  ssid: string
  country: string
  freq: number
  /** Omit to preserve the existing password. The host never returns it. */
  pass?: string
}

export async function fetchUltimateWifiConfig(): Promise<UltimateWifiConfig> {
  return json(
    await fetch(apiUrl("/api/tracker/ultimate/wifi"), {
      cache: "no-store",
    })
  )
}

export async function saveUltimateWifiConfig(
  update: UltimateWifiUpdate
): Promise<UltimateWifiConfig> {
  return json(
    await fetch(apiUrl("/api/tracker/ultimate/wifi"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    })
  )
}

export type TrackerCalibrationStatus = TrackerTransformStatus | "unbound"

export interface TrackerCalibrationSide {
  key: string | null
  status: TrackerCalibrationStatus
  /** Exact active and legacy/same-family entries eligible for explicit removal. */
  overrideKeys?: string[]
  /** Content revisions guarding confirmation-gated removal against stale tabs. */
  overrideRevisions?: Record<string, string>
  /** Measured and stale saved values are returned; factory/candidate values stay null. */
  pos: [number, number, number] | null
  quat: [number, number, number, number] | null
  /** Ultimate parser convention recorded with this measurement, when present. */
  poseConvention?: { quatOrder: "xyzw" | "wxyz"; upAxis: "y" | "z" } | null
}

export interface TrackerCalibrationSnapshot {
  path: string
  source: MantisTrackerSource
  keys: { left: string | null; right: string | null }
  /** Active Ultimate parser convention; absent/null for older hosts and other sources. */
  activePoseConvention?: { quatOrder: "xyzw" | "wxyz"; upAxis: "y" | "z" } | null
  left: TrackerCalibrationSide
  right: TrackerCalibrationSide
}

export interface TrackerCalibrationValue {
  /** Echoed active key; the host rejects the write if identification changed. */
  key: string
  pos: [number, number, number]
  quat: [number, number, number, number]
}

export type TrackerCalibrationUpdate = Partial<Record<"left" | "right", TrackerCalibrationValue>>

export async function fetchTrackerCalibration(
  source: MantisTrackerSource
): Promise<TrackerCalibrationSnapshot> {
  return json(
    await fetch(apiUrl(`/api/tracker/calibration/${source}`), {
      cache: "no-store",
    })
  )
}

export async function saveTrackerCalibration(
  source: MantisTrackerSource,
  update: TrackerCalibrationUpdate
): Promise<TrackerCalibrationSnapshot> {
  return json(
    await fetch(apiUrl(`/api/tracker/calibration/${source}`), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    })
  )
}

export async function removeTrackerCalibration(
  source: MantisTrackerSource,
  side: "left" | "right",
  key: string,
  activeKey: string,
  revision: string
): Promise<TrackerCalibrationSnapshot> {
  const query = new URLSearchParams({ key, active_key: activeKey, revision })
  return json(
    await fetch(apiUrl(`/api/tracker/calibration/${source}/${side}?${query.toString()}`), {
      method: "DELETE",
    })
  )
}

// ---------------------------------------------------------------------------
// Shared operator settings (serve/settings.py) — persisted on the serve host
// at ~/.almond/settings.json and folded into every op start server-side.
// ---------------------------------------------------------------------------

export type SettingValue = string | number | boolean | number[]

/** Optional widget hints for a settings field (slider ranges, pose editor,
 * toggle-number = a switch arming a numeric value where 0 means off). */
export interface SettingsFieldUI {
  widget?: "slider" | "pose" | "toggle-number"
  min?: number
  max?: number
  step?: number
  /** toggle-number: value filled in when the switch turns on. */
  onValue?: number
}

export interface SettingsField {
  key: string
  label: string
  type: FieldType
  help: string
  options: string[] | null
  /** Resolved from the op config dataclass defaults (null if unresolvable). */
  default: SettingValue | null
  /** What a null default actually does (e.g. the LeRobot cache dir), for the placeholder. */
  defaultText?: string | null
  ui: SettingsFieldUI
  /** op id -> dotted config keys this setting drives on that op. */
  targets: Record<string, string[]>
}

export interface SettingsCategory {
  key: string
  label: string
  description: string
  settings: SettingsField[]
}

/** One canonical subsystem in the unified Advanced tree (serve/settings.py).
 * Its values apply to every operation that has the subsystem. */
export interface AdvancedSection {
  key: string
  label: string
  nodes: SchemaNode[]
}

export interface SettingsSnapshot {
  /** Stored shared values keyed by canonical setting key (sparse: only set ones). */
  values: Record<string, SettingValue>
  /** Stored camera spec, or null when never configured on this host. */
  cameras: CameraSpec | null
  /** Advanced values keyed canonically (e.g. "axol.left.elbow.kp") — one
   * source of truth, translated to each op's config path server-side. */
  advanced: Record<string, FormValue>
  schema: SettingsCategory[]
  advancedSchema: AdvancedSection[]
}

export interface SettingsPatch {
  /** Per-key merge; null resets a key to its default. */
  values?: Record<string, SettingValue | null>
  cameras?: CameraSpec | null
  /** Must accompany `cameras: null` so clearing is distinguishable from omitting. */
  camerasSet?: boolean
  /** Per-key merge of canonical advanced values; null resets a key. */
  advanced?: Record<string, FormValue | null>
}

export async function fetchSettings(): Promise<SettingsSnapshot> {
  return json(await fetch(apiUrl("/api/settings")))
}

export async function saveSettings(patch: SettingsPatch): Promise<SettingsSnapshot> {
  const res: Omit<SettingsSnapshot, "schema" | "advancedSchema"> = await json(
    await fetch(apiUrl("/api/settings"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    })
  )
  return { schema: [], advancedSchema: [], ...res }
}

/** URL of the robot's URDF (meshes resolve relative to it via /api/urdf/…). */
export function urdfUrl(): string {
  return apiUrl("/api/urdf/axol.urdf")
}

export function cameraSerials(
  spec: CameraSpec,
  mantis?: boolean
): Partial<Record<CameraSlot, string>> {
  if (mantis === true) {
    return (
      spec.mantis_serials ?? {
        left_arm: spec.serials.left_arm,
        right_arm: spec.serials.right_arm,
      }
    )
  }
  return spec.serials
}

export function cameraCount(spec: CameraSpec, mantis?: boolean): number {
  const values =
    mantis === undefined
      ? [...Object.values(spec.serials), ...Object.values(spec.mantis_serials ?? {})]
      : Object.values(cameraSerials(spec, mantis))
  return new Set(values.map((s) => s?.trim()).filter(Boolean)).size
}

function branchResolutionEnabled(spec: CameraSpec, branch: "stream" | "record"): boolean {
  const value =
    branch === "stream"
      ? (spec.stream_resolution ?? spec.resolution ?? "SVGA")
      : (spec.record_resolution ?? "SVGA")
  return value.trim() !== "" && value.toLowerCase() !== RESOLUTION_OFF
}

function cameraBranchEnabled(
  spec: CameraSpec,
  slot: CameraSlot,
  branch: "stream" | "record"
): boolean {
  if (!branchResolutionEnabled(spec, branch)) return false
  const raw: unknown = spec[branch]?.[slot]
  return raw !== false && raw !== null && raw !== "off" && raw !== ""
}

/** Assigned serials that an operation will actually open. */
export function participatingCameraSerials(
  spec: CameraSpec,
  mantis = false,
  branches: { stream: boolean; record: boolean } = { stream: true, record: true }
): string[] {
  const values = Object.entries(cameraSerials(spec, mantis))
    .filter(
      ([slot]) =>
        (branches.stream && cameraBranchEnabled(spec, slot as CameraSlot, "stream")) ||
        (branches.record && cameraBranchEnabled(spec, slot as CameraSlot, "record"))
    )
    .map(([, serial]) => serial?.trim() ?? "")
    .filter(Boolean)
  return [...new Set(values)]
}

/** Number of distinct assigned cameras included in the dataset branch. */
export function recordingCameraCount(spec: CameraSpec, mantis = false): number {
  return participatingCameraSerials(spec, mantis, { stream: false, record: true }).length
}

/**
 * Configured camera serials that are NOT among the detected ZED devices — i.e.
 * cameras the operator assigned but that aren't physically connected. An empty
 * result means every assigned camera was found.
 */
export function missingCameraSerials(
  spec: CameraSpec,
  detected: CameraDevice[],
  mantis = false,
  branches: { stream: boolean; record: boolean } = { stream: true, record: true }
): string[] {
  const present = new Set(detected.map((d) => String(d.serial)))
  return participatingCameraSerials(spec, mantis, branches).filter((s) => !present.has(s))
}

// ---------------------------------------------------------------------------
// Schema helpers
// ---------------------------------------------------------------------------

export function flattenFields(nodes: SchemaNode[]): SchemaField[] {
  const out: SchemaField[] = []
  for (const node of nodes) {
    if (node.kind === "group") out.push(...flattenFields(node.children))
    else out.push(node)
  }
  return out
}

/**
 * Prune a schema tree, dropping leaf fields whose key is in `exclude` (and any
 * groups left empty as a result). Used to render "everything else" beneath an
 * operation's curated common fields without showing them twice.
 */
export function filterSchema(nodes: SchemaNode[], exclude: Set<string>): SchemaNode[] {
  const out: SchemaNode[] = []
  for (const node of nodes) {
    if (node.kind === "field") {
      if (!exclude.has(node.key)) out.push(node)
    } else {
      const children = filterSchema(node.children, exclude)
      if (children.length > 0) out.push({ ...node, children })
    }
  }
  return out
}

export function defaultString(field: SchemaField): string {
  return field.default == null ? "" : String(field.default)
}

/** Has the user changed this field away from its default? */
export function isModified(field: SchemaField, value: FormValue | undefined): boolean {
  if (value === undefined) return false
  if (field.type === "boolean") return Boolean(value) !== Boolean(field.default)
  return String(value) !== defaultString(field)
}

/** Required fields with no value supplied yet (blocks Start). */
export function missingRequired(
  fields: SchemaField[],
  overrides: Record<string, FormValue>
): string[] {
  return fields
    .filter((f) => f.required)
    .filter((f) => {
      const v = overrides[f.key]
      return v === undefined || String(v).trim() === ""
    })
    .map((f) => f.key)
}

/**
 * The minimal args to send: every required field, plus any field the user
 * changed from its default. Unchanged optional fields are omitted so the
 * command falls back to its own defaults.
 */
export function computeArgs(
  fields: SchemaField[],
  overrides: Record<string, FormValue>
): Record<string, FormValue> {
  const args: Record<string, FormValue> = {}
  for (const field of fields) {
    const has = field.key in overrides
    if (field.required) {
      args[field.key] = has ? overrides[field.key] : ""
    } else if (has && isModified(field, overrides[field.key])) {
      args[field.key] = overrides[field.key]
    }
  }
  return args
}

// ---------------------------------------------------------------------------
// Operations. The connected serve host advertises which ones exist and what
// each one needs (/api/commands), so a host with extra operations registered
// gets panels for them here without a frontend change.
//
// Everything tunable-but-stable (stiffness, rates, codecs, the inference
// server, …) lives in the shared Settings dialog and is folded in server-side;
// the op panels only ask for what changes run to run. Field keys are the dotted
// draccus paths the backend understands (serve/commands.py build_argv).
// ---------------------------------------------------------------------------

export interface OperationMeta {
  id: OperationId
  label: string
  description: string
  /** Per-run config keys surfaced in the panel (required + run identity). */
  fields: string[]
  /** Needs the persistent robot connection (CAN) to run. */
  requiresRobot: boolean
  /** Needs at least one camera serial configured (collect-data / run-policy). */
  requiresCameras: boolean
  /** Can run in sim (no hardware) — only teleop today. */
  simCapable: boolean
  /** Arg that makes a run hardware-free; null when the robot is required. */
  simFlag: string | null
  /** Args that skip the arm-robot gates without being sim (teleop's cart_only:
   * real hardware, but the arms and their CAN bus are never touched). */
  robotFreeFlags: string[]
  /** Runtime supports the Mantis hardware profile. */
  supportsMantis: boolean
  /** Shows the episode start / save / discard controls while running. */
  episodeControl: boolean
  /** Shows the "point the headset at this machine" hint while running. */
  usesHeadset: boolean
  /** Honors cameras selected for headset streaming, in addition to recording. */
  streamsVideo: boolean
}

/**
 * The operations every serve host has had since before the command registry.
 * Used verbatim when the host's /api/commands predates it, so an up-to-date
 * panel still drives an older robot.
 */
export const OPERATIONS: OperationMeta[] = [
  {
    id: "teleop",
    label: "Teleoperation",
    description: "Drive Axol from VR; Mantis supports Quest, Lighthouse, or Ultimate tracking.",
    fields: ["sim"],
    requiresRobot: true,
    requiresCameras: false,
    simCapable: true,
    simFlag: "sim",
    robotFreeFlags: ["mantis"],
    supportsMantis: true,
    episodeControl: false,
    usesHeadset: true,
    streamsVideo: true,
  },
  {
    id: "gravity-comp",
    label: "Gravity compensation",
    description: "Hold the arms weightless so they can be moved by hand.",
    fields: ["free_joints"],
    requiresRobot: true,
    requiresCameras: false,
    simCapable: false,
    simFlag: null,
    robotFreeFlags: [],
    supportsMantis: false,
    episodeControl: false,
    usesHeadset: false,
    streamsVideo: false,
  },
  {
    id: "collect-data",
    label: "Collect data",
    description:
      "Record with ZED cameras; Mantis supports Quest, Lighthouse, or Ultimate tracking.",
    fields: ["repo_id", "task"],
    requiresRobot: true,
    requiresCameras: true,
    simCapable: false,
    simFlag: null,
    robotFreeFlags: ["mantis"],
    supportsMantis: true,
    // Panel-driven episodes are newer than the registry, so a host old enough
    // to need this table can't serve them — the controls would sit on
    // "Preparing" forever. Its collect-data does run the VR server with the
    // camera tracks, so the feeds are safe to offer.
    episodeControl: false,
    usesHeadset: true,
    streamsVideo: true,
  },
  {
    id: "replay-dataset",
    label: "Replay dataset",
    description: "Replay a recorded episode of a LeRobot dataset on Axol, then return to rest.",
    fields: ["repo_id", "episode", "loop", "interpolate"],
    requiresRobot: true,
    requiresCameras: false,
    simCapable: false,
    simFlag: null,
    robotFreeFlags: [],
    supportsMantis: false,
    episodeControl: false,
    usesHeadset: false,
    streamsVideo: false,
  },
  {
    id: "run-policy",
    label: "Run policy",
    description:
      "Run a trained policy on Axol via LeRobot async inference, locally or on a remote inference server.",
    fields: ["policy_path", "policy_type", "task", "repo_id"],
    requiresRobot: true,
    requiresCameras: true,
    simCapable: false,
    simFlag: null,
    robotFreeFlags: [],
    supportsMantis: false,
    episodeControl: true,
    usesHeadset: false,
    streamsVideo: false,
  },
]

/**
 * The operations the connected host offers, in catalog order.
 *
 * A host that predates the command registry answers /api/commands without the
 * operation fields; there is then nothing to read them from, so fall back to
 * the built-in list above rather than rendering an empty panel.
 */
export function operationsFromCommands(specs: CommandSpec[]): OperationMeta[] {
  const ops = specs.filter((s) => s.isOperation)
  if (ops.length === 0) return OPERATIONS
  return ops.map((s) => ({
    id: s.id,
    label: s.label,
    description: s.description,
    fields: s.perRunFields ?? [],
    // Every in-process operation drives the arms; only a sim run doesn't, and
    // that's decided per run from simFlag.
    requiresRobot: true,
    requiresCameras: Boolean(s.requiresCameras),
    simCapable: s.simCapable,
    simFlag: s.simFlag ?? null,
    robotFreeFlags: s.robotFreeFlags ?? [],
    supportsMantis: s.supportsMantis ?? Boolean(s.perRunFields?.includes("mantis")),
    episodeControl: Boolean(s.episodeControl),
    usesHeadset: Boolean(s.usesHeadset),
    streamsVideo: Boolean(s.streamsVideo),
  }))
}

/** Whether this run is hardware-free (so it skips the robot / fault gates). */
export function isSimRun(meta: OperationMeta, settings: Record<string, FormValue>): boolean {
  return meta.simFlag != null && Boolean(settings[meta.simFlag])
}

/**
 * Whether this run leaves the arms (and their CAN bus) untouched — sim, or a
 * robot-free flag like teleop's cart_only. Such a run skips the "Connect
 * Axol" and motor-fault gates; cart_only still drives real cart hardware.
 */
export function isRobotFreeRun(meta: OperationMeta, settings: Record<string, FormValue>): boolean {
  if (isSimRun(meta, settings)) return true
  return meta.robotFreeFlags.some((flag) => Boolean(settings[flag]))
}

/** Curated fields for an op, resolved from the introspected command schema. */
export function curatedFields(spec: CommandSpec, meta: OperationMeta): SchemaField[] {
  const byKey = new Map(flattenFields(spec.schema).map((f) => [f.key, f]))
  return meta.fields.map((k) => byKey.get(k)).filter((f): f is SchemaField => f != null)
}

/**
 * The fields an op panel shows (and the only args a start sends): the curated
 * per-run fields plus every required field, required first. Everything else
 * comes from the shared settings, folded in server-side. The device choice is
 * system-wide, so its flag (and, on Mantis, the Axol-only run modes) is never
 * one of them — even against a host that still lists it per run.
 */
export function perRunFields(
  spec: CommandSpec,
  meta: OperationMeta,
  profile: HardwareProfile = "axol"
): SchemaField[] {
  const byKey = new Map(curatedFields(spec, meta).map((f) => [f.key, f]))
  for (const f of flattenFields(spec.schema)) {
    if (f.required && !byKey.has(f.key)) byKey.set(f.key, f)
  }
  return [...byKey.values()]
    .filter((f) => runFieldVisible(f.key, profile))
    .sort((a, b) => Number(b.required) - Number(a.required))
}

// ---------------------------------------------------------------------------
// Per-operation settings: localStorage persistence + JSON import/export
// ---------------------------------------------------------------------------

const OP_SETTINGS_PREFIX = "axolOp:"

export function loadOpSettings(op: OperationId): Record<string, FormValue> {
  try {
    const raw = localStorage.getItem(`${OP_SETTINGS_PREFIX}${op}`)
    if (raw) {
      const parsed = JSON.parse(raw) as Record<string, FormValue>
      // The Axol/Mantis choice used to be a per-operation toggle stored here.
      // It is now the system-wide device selection, so a stale flag must not
      // linger as an "edited" per-run value.
      delete parsed[HARDWARE_PROFILE_ARG]
      return parsed
    }
  } catch {
    // ignore malformed storage
  }
  return {}
}

export function saveOpSettings(op: OperationId, settings: Record<string, FormValue>): void {
  try {
    localStorage.setItem(`${OP_SETTINGS_PREFIX}${op}`, JSON.stringify(settings))
  } catch {
    // ignore storage failures (private mode / quota)
  }
}

// ---------------------------------------------------------------------------
// Log streaming
// ---------------------------------------------------------------------------

function wsUrl(id: string): string {
  return `${wsBaseUrl()}/api/sessions/${id}/logs`
}

interface LogMessage {
  type: "log" | "status" | "error"
  line?: string
  message?: string
  session?: SessionInfo
}

/** Streams a session's log lines and live status over a WebSocket. */
export function useSessionLogs(sessionId: string | null): {
  lines: string[]
  status: SessionInfo | null
} {
  const [logState, setLogState] = useState<{
    sessionId: string | null
    lines: string[]
    status: SessionInfo | null
  }>({ sessionId: null, lines: [], status: null })
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!sessionId) return

    const ws = new WebSocket(wsUrl(sessionId))
    wsRef.current = ws

    ws.onmessage = (event) => {
      const msg: LogMessage = JSON.parse(event.data)
      setLogState((previous) => {
        const current =
          previous.sessionId === sessionId ? previous : { sessionId, lines: [], status: null }
        if (msg.type === "log" && msg.line !== undefined) {
          const base =
            current.lines.length >= MAX_LINES ? current.lines.slice(-MAX_LINES + 1) : current.lines
          return { ...current, lines: [...base, msg.line] }
        }
        if (msg.type === "status" && msg.session) {
          return { ...current, status: msg.session }
        }
        if (msg.type === "error" && msg.message) {
          return { ...current, lines: [...current.lines, `[error] ${msg.message}`] }
        }
        return current
      })
    }

    return () => {
      ws.onmessage = null
      ws.close()
      wsRef.current = null
    }
  }, [sessionId])

  return logState.sessionId === sessionId
    ? { lines: logState.lines, status: logState.status }
    : { lines: [], status: null }
}
