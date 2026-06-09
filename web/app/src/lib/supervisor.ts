import { useEffect, useRef, useState } from "react"

export type FieldType = "boolean" | "number" | "select" | "text"

/** A single configurable leaf in a command's config (serve/introspect.py). */
export interface SchemaField {
  kind: "field"
  key: string
  label: string
  type: FieldType
  default: string | number | boolean | null
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
}

/** Catalog category display order (matches serve/commands.py CATEGORY_ORDER). */
export const CATEGORY_ORDER = ["Operate", "Cameras", "Calibrate", "Setup"]

export type SessionStatus = "starting" | "running" | "exited" | "error"

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

export type FormValue = string | boolean

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

function apiUrl(path: string): string {
  return `${apiBase}${path}`
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export interface ServerInfo {
  hostname: string
  lanIp: string
  viewerPort: number
  vrPort: number
  /** Best-guess wired ZED-link interface on the serve host (Linux only). */
  ethIface: string | null
  /** All plausible wired interfaces, best candidate first. */
  ethIfaces: string[]
}

export async function fetchInfo(): Promise<ServerInfo> {
  return json(await fetch(apiUrl("/api/info")))
}

/** Reach the ZED box's own `axol serve` (proxied) to validate + list ifaces. */
export async function fetchBoxInfo(url: string): Promise<ServerInfo> {
  return json(await fetch(apiUrl(`/api/zed/box-info?url=${encodeURIComponent(url)}`)))
}

// ---------------------------------------------------------------------------
// Robot connection (detached CAN + 1 Hz motor ping)
// ---------------------------------------------------------------------------

export type RobotState = "disconnected" | "connecting" | "connected" | "busy" | "error"

export interface MotorHealth {
  arm: string
  joint: string
  reachable: boolean
  status: string | null
}

export interface RobotStatus {
  state: RobotState
  connected: boolean
  error: string | null
  needsSudo: boolean
  lastPing: number | null
  motors: MotorHealth[]
  motorCount: number
  reachableCount: number
}

export async function fetchRobotStatus(): Promise<RobotStatus> {
  return json(await fetch(apiUrl("/api/robot/status")))
}

export async function robotConnect(password?: string): Promise<RobotStatus> {
  return json(
    await fetch(apiUrl("/api/robot/connect"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: password ?? null }),
    })
  )
}

export async function robotDisconnect(): Promise<RobotStatus> {
  return json(await fetch(apiUrl("/api/robot/disconnect"), { method: "POST" }))
}

// ---------------------------------------------------------------------------
// ZED box link (detached, lightweight reachability check)
// ---------------------------------------------------------------------------

export interface PtpStatus {
  running: boolean
  locked: boolean
  offsetNs: number | null
  sessionId: string | null
  needsSudo?: boolean
  badPassword?: boolean
  error?: string | null
}

export interface StreamStatus {
  streaming: boolean
  ready: boolean
  cameras: string[]
  sessionId: string | null
  error?: string | null
}

export interface ZedLinkStatus {
  connected: boolean
  boxUrl: string | null
  info: ServerInfo | null
  error: string | null
  ptp?: PtpStatus
  stream?: StreamStatus
}

export async function fetchZedStatus(): Promise<ZedLinkStatus> {
  return json(await fetch(apiUrl("/api/zed/status")))
}

export async function zedConnect(
  url: string,
  password?: string,
  cameras?: ZedSpec["cameras"]
): Promise<ZedLinkStatus> {
  const body: { url: string; password?: string; cameras?: ZedSpec["cameras"] } = { url }
  if (password) body.password = password
  if (cameras) body.cameras = cameras
  return json(
    await fetch(apiUrl("/api/zed/connect"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
  )
}

export async function zedDisconnect(): Promise<ZedLinkStatus> {
  return json(await fetch(apiUrl("/api/zed/disconnect"), { method: "POST" }))
}

// ---------------------------------------------------------------------------
// In-process operations (teleop / gravity-comp / collect-data / run-policy)
// ---------------------------------------------------------------------------

export type OperationId = "teleop" | "gravity-comp" | "collect-data" | "run-policy"

export interface OpStatus {
  running: boolean
  session: SessionInfo | null
}

export async function fetchOpStatus(): Promise<OpStatus> {
  return json(await fetch(apiUrl("/api/op/status")))
}

export async function startOperation(
  op: OperationId,
  args: Record<string, FormValue>,
  zed?: ZedSpec
): Promise<SessionInfo> {
  return json(
    await fetch(apiUrl("/api/op/start"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ op, args, zed: zed ?? null }),
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

/** Serve-side orchestration spec for collect-data / run-policy (see orchestrator.py).
 *
 * The cameras stream from, and the clocks are PTP-synced over, the same ZED box
 * address ``axol serve`` is reachable on; the PTP interfaces on both machines
 * are auto-derived from it server-side. Network addressing between this host
 * and the box is the operator's job.
 */
export interface ZedSpec {
  enabled: boolean
  boxUrl: string
  cameras: { overhead: string; left_arm: string; right_arm: string }
  resolution?: string
  fps?: number
  bitrate?: number
}

export async function fetchCommands(): Promise<CommandSpec[]> {
  return json(await fetch(apiUrl("/api/commands")))
}

export async function fetchSessions(): Promise<SessionInfo[]> {
  return json(await fetch(apiUrl("/api/sessions")))
}

export async function runCommand(
  command: string,
  args: Record<string, FormValue>,
  zed?: ZedSpec
): Promise<SessionInfo> {
  return json(
    await fetch(apiUrl("/api/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, args, zed: zed ?? null }),
    })
  )
}

export async function stopSession(id: string): Promise<SessionInfo> {
  return json(await fetch(apiUrl(`/api/sessions/${id}/stop`), { method: "POST" }))
}

export function zedCameraCount(spec: ZedSpec): number {
  return Object.values(spec.cameras).filter((s) => s.trim()).length
}

/** Human-readable list of ZED fields blocking Start (empty = ready). */
export function zedMissing(spec: ZedSpec): string[] {
  if (!spec.enabled) return []
  const missing: string[] = []
  if (!spec.boxUrl.trim()) missing.push("ZED box address")
  if (zedCameraCount(spec) === 0) missing.push("at least one camera serial")
  return missing
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
// Curated operations: the friendly subset of fields each op panel shows.
// Keys are the dotted draccus paths the backend understands (serve/commands.py
// build_argv); unlisted fields fall back to their config defaults.
// ---------------------------------------------------------------------------

export interface OperationMeta {
  id: OperationId
  label: string
  description: string
  /** Curated config keys surfaced in the panel (others use their defaults). */
  fields: string[]
  /** Needs the persistent robot connection (CAN) to run. */
  requiresRobot: boolean
  /** Needs the ZED box link (collect-data / run-policy). */
  requiresZed: boolean
  /** Can run in sim (no hardware) — only teleop today. */
  simCapable: boolean
}

export const OPERATIONS: OperationMeta[] = [
  {
    id: "teleop",
    label: "Teleoperation",
    description: "Drive the Axol from a VR headset. Enable sim to preview in the browser.",
    fields: ["sim", "teleop.frequency", "axol.left_stiffness", "axol.right_stiffness"],
    requiresRobot: true,
    requiresZed: false,
    simCapable: true,
  },
  {
    id: "gravity-comp",
    label: "Gravity Compensation",
    description: "Hold the arms weightless so they can be moved by hand.",
    fields: ["kd", "rate_hz", "free_joints"],
    requiresRobot: true,
    requiresZed: false,
    simCapable: false,
  },
  {
    id: "collect-data",
    label: "Collect Data",
    description: "Record teleoperation episodes to a LeRobot dataset with the ZED cameras.",
    fields: ["repo_id", "task", "fps", "push_to_hub"],
    requiresRobot: true,
    requiresZed: true,
    simCapable: false,
  },
  {
    id: "run-policy",
    label: "Run Policy",
    description: "Run a trained policy on Axol via LeRobot async inference.",
    fields: ["policy_path", "repo_id", "task", "episode_time_s", "server_port"],
    requiresRobot: true,
    requiresZed: true,
    simCapable: false,
  },
]

export function operationMeta(op: OperationId): OperationMeta {
  return OPERATIONS.find((o) => o.id === op) as OperationMeta
}

/** Curated fields for an op, resolved from the introspected command schema. */
export function curatedFields(spec: CommandSpec, meta: OperationMeta): SchemaField[] {
  const byKey = new Map(flattenFields(spec.schema).map((f) => [f.key, f]))
  return meta.fields.map((k) => byKey.get(k)).filter((f): f is SchemaField => f != null)
}

// ---------------------------------------------------------------------------
// Per-operation settings: localStorage persistence + JSON import/export
// ---------------------------------------------------------------------------

const OP_SETTINGS_PREFIX = "axolOp:"

export function loadOpSettings(op: OperationId): Record<string, FormValue> {
  try {
    const raw = localStorage.getItem(`${OP_SETTINGS_PREFIX}${op}`)
    if (raw) return JSON.parse(raw) as Record<string, FormValue>
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

/** Trigger a browser download of an operation's settings as JSON. */
export function exportOpSettings(op: OperationId, settings: Record<string, FormValue>): void {
  const blob = new Blob([JSON.stringify({ op, settings }, null, 2)], {
    type: "application/json",
  })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `${op}-settings.json`
  a.click()
  URL.revokeObjectURL(url)
}

/** Parse an imported settings file; accepts `{op, settings}` or a bare map. */
export function parseImportedSettings(text: string): Record<string, FormValue> {
  const data = JSON.parse(text)
  const settings = data && typeof data === "object" && "settings" in data ? data.settings : data
  if (!settings || typeof settings !== "object") throw new Error("invalid settings file")
  const out: Record<string, FormValue> = {}
  for (const [k, v] of Object.entries(settings as Record<string, unknown>)) {
    if (typeof v === "string" || typeof v === "boolean" || typeof v === "number") {
      out[k] = typeof v === "number" ? String(v) : v
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// Log streaming
// ---------------------------------------------------------------------------

function wsUrl(id: string): string {
  const base = apiBase || window.location.origin
  const u = new URL(base)
  const proto = u.protocol === "https:" ? "wss" : "ws"
  return `${proto}://${u.host}/api/sessions/${id}/logs`
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
  const [lines, setLines] = useState<string[]>([])
  const [status, setStatus] = useState<SessionInfo | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    setLines([])
    setStatus(null)
    if (!sessionId) return

    const ws = new WebSocket(wsUrl(sessionId))
    wsRef.current = ws

    ws.onmessage = (event) => {
      const msg: LogMessage = JSON.parse(event.data)
      if (msg.type === "log" && msg.line !== undefined) {
        setLines((prev) => {
          const base = prev.length >= MAX_LINES ? prev.slice(-MAX_LINES + 1) : prev
          return [...base, msg.line as string]
        })
      } else if (msg.type === "status" && msg.session) {
        setStatus(msg.session)
      } else if (msg.type === "error" && msg.message) {
        setLines((prev) => [...prev, `[error] ${msg.message}`])
      }
    }

    return () => {
      ws.onmessage = null
      ws.close()
      wsRef.current = null
    }
  }, [sessionId])

  return { lines, status }
}
