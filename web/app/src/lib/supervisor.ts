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
  simCapable: boolean
  requiresHardware: boolean
  available: boolean
  error: string | null
  schema: SchemaNode[]
  required: string[]
}

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
  return json(await fetch("/api/info"))
}

/** Reach the ZED box's own `axol serve` (proxied) to validate + list ifaces. */
export async function fetchBoxInfo(url: string): Promise<ServerInfo> {
  return json(await fetch(`/api/zed/box-info?url=${encodeURIComponent(url)}`))
}

export type ZedTopology = "direct" | "lan"

/** Serve-side orchestration spec for collect-data / run-policy (see orchestrator.py). */
export interface ZedSpec {
  enabled: boolean
  boxUrl: string
  topology: ZedTopology
  hostIface: string
  boxIface: string
  zedHost: string
  cameras: { overhead: string; left_arm: string; right_arm: string }
  resolution?: string
  fps?: number
  bitrate?: number
}

export async function fetchCommands(): Promise<CommandSpec[]> {
  return json(await fetch("/api/commands"))
}

export async function fetchSessions(): Promise<SessionInfo[]> {
  return json(await fetch("/api/sessions"))
}

export async function runCommand(
  command: string,
  args: Record<string, FormValue>,
  zed?: ZedSpec
): Promise<SessionInfo> {
  return json(
    await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, args, zed: zed ?? null }),
    })
  )
}

export async function stopSession(id: string): Promise<SessionInfo> {
  return json(await fetch(`/api/sessions/${id}/stop`, { method: "POST" }))
}

export function zedCameraCount(spec: ZedSpec): number {
  return Object.values(spec.cameras).filter((s) => s.trim()).length
}

/** Human-readable list of ZED fields blocking Start (empty = ready). */
export function zedMissing(spec: ZedSpec): string[] {
  if (!spec.enabled) return []
  const missing: string[] = []
  if (!spec.boxUrl.trim()) missing.push("ZED box address")
  if (!spec.hostIface.trim()) missing.push("host interface")
  if (!spec.boxIface.trim()) missing.push("ZED box interface")
  if (spec.topology === "lan" && !spec.zedHost.trim()) missing.push("ZED stream IP")
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
// Log streaming
// ---------------------------------------------------------------------------

function wsUrl(id: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws"
  return `${proto}://${window.location.host}/api/sessions/${id}/logs`
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
