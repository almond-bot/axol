import { useEffect, useState } from "react"
import { apiUrl, wsBaseUrl, type RobotState, type SessionInfo } from "@/lib/supervisor"

// ---------------------------------------------------------------------------
// Motor telemetry: types + fetchers + live WebSocket stream
// (server side: almond_axol/serve/telemetry.py)
// ---------------------------------------------------------------------------

/** Joints in control order — fixed chart-series slots (color follows the joint). */
export const JOINTS = [
  "SHOULDER_1",
  "SHOULDER_2",
  "SHOULDER_3",
  "ELBOW",
  "WRIST_1",
  "WRIST_2",
  "WRIST_3",
  "GRIPPER",
] as const

export type JointName = (typeof JOINTS)[number]
export type ArmSide = "left" | "right"

/**
 * Categorical palette for the eight joints, one fixed slot per joint (never
 * reassigned when series are filtered). Validated for the dark card surface
 * (#161618): all slots ≥3:1 contrast; adjacent-slot CVD separation sits in the
 * 8–12 floor band, so charts always pair the colors with a legend + per-series
 * value labels.
 */
export const JOINT_COLORS: Record<JointName, string> = {
  SHOULDER_1: "#3987e5",
  SHOULDER_2: "#199e70",
  SHOULDER_3: "#c98500",
  ELBOW: "#008300",
  WRIST_1: "#9085e9",
  WRIST_2: "#e66767",
  WRIST_3: "#d55181",
  GRIPPER: "#d95926",
}

export function jointLabel(joint: string): string {
  return joint.toLowerCase().replace("_", " ")
}

export function motorKey(arm: ArmSide, joint: JointName): string {
  return `${arm}:${joint}`
}

/** One fast sample per motor: [position rad, velocity rad/s, torque Nm].
 * Entries may be null in run captures (a field the capture didn't record). */
export type FastSample = (number | null)[]

export interface TelemetryFrame {
  t: number
  m: Record<string, FastSample>
}

/** Slow (1 Hz) per-motor reading piggybacked on the health ping. */
export interface SlowReading {
  reachable: boolean
  status: string | null
  temperature: number | null
  voltage: number | null
}

export interface TelemetrySnapshot {
  state: RobotState
  sampleHz: number
  slow: Record<string, SlowReading>
  slowT: number | null
  latest: TelemetryFrame | null
}

export async function fetchTelemetry(): Promise<TelemetrySnapshot> {
  return json(await fetch(apiUrl("/api/telemetry")))
}

export async function fetchTelemetryHistory(
  seconds: number,
  maxFrames = 2000
): Promise<{ frames: TelemetryFrame[] }> {
  return json(
    await fetch(apiUrl(`/api/telemetry/history?seconds=${seconds}&max_frames=${maxFrames}`))
  )
}

/** Full one-motor readout (the `motor.info` set) over the idle robot link. */
export interface MotorDetails {
  arm: string
  joint: string
  model: string | null
  firmware: number | null
  status: string | null
  mode: string | null
  position: number | null
  velocity: number | null
  torque: number | null
  temperature: number | null
  voltage: number | null
  gains: Record<string, number | null> | null
}

export async function fetchMotorDetails(arm: string, joint: string): Promise<MotorDetails> {
  return json(await fetch(apiUrl(`/api/robot/motors/${arm}/${joint}`)))
}

// ---------------------------------------------------------------------------
// Diagnostics runs (script launches wrapped with telemetry capture)
// ---------------------------------------------------------------------------

export interface DiagnosticsRunMeta {
  id: string
  sessionId: string
  command: string
  args: Record<string, unknown>
  startedAt: number
  endedAt: number | null
  status: string
  exitCode: number | null
  telemetryCsv?: string | null
  frameCount?: number
}

export interface DiagnosticsRunData {
  meta: DiagnosticsRunMeta
  frames: TelemetryFrame[]
  log: string[]
}

export async function startDiagnosticsRun(
  command: string,
  args: Record<string, unknown>
): Promise<{ run: DiagnosticsRunMeta; session: SessionInfo }> {
  return json(
    await fetch(apiUrl("/api/diagnostics/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, args }),
    })
  )
}

export async function fetchDiagnosticsRuns(): Promise<{ runs: DiagnosticsRunMeta[] }> {
  return json(await fetch(apiUrl("/api/diagnostics/runs")))
}

export async function fetchDiagnosticsRun(id: string): Promise<DiagnosticsRunData> {
  return json(await fetch(apiUrl(`/api/diagnostics/runs/${id}`)))
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Live stream hook
// ---------------------------------------------------------------------------

/** Client-side frame buffer: 10 minutes at the server's 10 Hz sample rate. */
const MAX_FRAMES = 6000
const RECONNECT_MS = 3000

type WsMessage =
  | ({ type: "hello" } & TelemetrySnapshot)
  | ({ type: "frame" } & TelemetryFrame)
  | { type: "slow"; t: number; m: Record<string, SlowReading> }
  | { type: "state"; state: RobotState }

export interface TelemetryStream {
  /** Robot-link state as reported over the stream (why frames stop coming). */
  state: RobotState
  /** Latest slow reading per motor key ("left:ELBOW"). */
  slow: Record<string, SlowReading>
  /** Shared, append-only frame buffer (same array identity across renders). */
  frames: TelemetryFrame[]
  /** Bumped on every appended frame — subscribe charts to this. */
  version: number
  /** WebSocket currently open. */
  streaming: boolean
}

/**
 * Subscribes to `/api/telemetry/ws`, backfills history on (re)connect, and
 * keeps a capped rolling frame buffer. The buffer array is mutated in place
 * (stable identity) and `version` ticks per frame so consumers can redraw
 * without re-allocating 6k frames per render.
 */
export function useTelemetryStream(enabled: boolean): TelemetryStream {
  // The buffer is intentionally a stable, in-place-mutated array (returning a
  // new 6k-frame array per sample would thrash); `version` signals changes.
  const [frames] = useState<TelemetryFrame[]>(() => [])
  const [version, setVersion] = useState(0)
  const [state, setState] = useState<RobotState>("disconnected")
  const [slow, setSlow] = useState<Record<string, SlowReading>>({})
  const [streaming, setStreaming] = useState(false)

  useEffect(() => {
    if (!enabled) return
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const buf = frames
    const append = (frame: TelemetryFrame) => {
      buf.push(frame)
      if (buf.length > MAX_FRAMES) buf.splice(0, buf.length - MAX_FRAMES)
      setVersion((v) => v + 1)
    }

    const backfill = async () => {
      try {
        const { frames: history } = await fetchTelemetryHistory(600, 3000)
        const newest = buf.length > 0 ? buf[buf.length - 1].t : -Infinity
        // Only frames older than what we already streamed, to keep t monotonic.
        const older = history.filter((f) => f.t < newest || buf.length === 0)
        buf.splice(0, 0, ...older)
        if (buf.length > MAX_FRAMES) buf.splice(0, buf.length - MAX_FRAMES)
        setVersion((v) => v + 1)
      } catch {
        // history is best-effort; the live stream still works without it
      }
    }

    const connect = () => {
      if (closed) return
      ws = new WebSocket(`${wsBaseUrl()}/api/telemetry/ws`)
      ws.onopen = () => {
        setStreaming(true)
        backfill()
      }
      ws.onmessage = (event) => {
        const msg: WsMessage = JSON.parse(event.data)
        if (msg.type === "frame") {
          append({ t: msg.t, m: msg.m })
        } else if (msg.type === "slow") {
          setSlow((prev) => ({ ...prev, ...msg.m }))
        } else if (msg.type === "state") {
          setState(msg.state)
        } else if (msg.type === "hello") {
          setState(msg.state)
          setSlow(msg.slow)
        }
      }
      ws.onclose = () => {
        setStreaming(false)
        if (!closed) retry = setTimeout(connect, RECONNECT_MS)
      }
      ws.onerror = () => ws?.close()
    }
    connect()

    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      if (ws) {
        ws.onclose = null
        ws.close()
      }
      setStreaming(false)
    }
  }, [enabled, frames])

  return { state, slow, frames, version, streaming }
}
