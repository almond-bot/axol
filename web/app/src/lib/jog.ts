/**
 * Client for the jog session (`axol serve`'s `jog` operation).
 *
 * The page plans and executes arm moves over REST and streams hold commands
 * (base velocity, lift direction) over `/api/jog/ws`, which also pushes the
 * live session state. Poses are in Axol's torso frame: +x forward, +y left,
 * +z up, metres, at the gripper tips.
 */
import { useEffect, useRef, useState } from "react"
import { apiUrl, wsBaseUrl } from "@/lib/supervisor"

export type JogArm = "left" | "right"
export type Vec3 = [number, number, number]
/** Quaternion as `[x, y, z, w]`. */
export type Quat = [number, number, number, number]

export interface JogPose {
  position: Vec3
  quaternion: Quat
}

export interface JogLiftState {
  /** Height as a fraction of travel, 1 = fully raised; null until homed. */
  raised: number | null
  /** Fractions of travel per second, positive up. */
  velocity: number
  moving: boolean
}

export interface JogBaseState {
  x: number
  y: number
  yaw: number
  /** Normalized command the wheels are running. */
  command: Vec3
  /** Body-frame velocity, m/s and rad/s. */
  velocity: Vec3
  moving: boolean
  /** True on hardware, where the pose is integrated from the wheel command. */
  estimated: boolean
  fault?: boolean
}

export interface JogState {
  running: boolean
  mode?: "sim" | "hardware"
  ready?: boolean
  solverError?: string | null
  hasArms?: boolean
  arms?: { left: number[]; right: number[] } | null
  tips?: { left: JogPose; right: JogPose } | null
  lift?: JogLiftState | null
  base?: JogBaseState | null
  hold?: { active: boolean; drive: Vec3; lift: number }
  execution?: { id: number; arm: JogArm; progress: number } | null
  lastExecution?: { id: number; arm: JogArm; status: "done" | "stopped" } | null
}

export interface JogPlan {
  id: number
  arm: JogArm
  target: JogPose
  reached: JogPose
  reachable: boolean
  positionError: number
  orientationError: number
  joints: number[]
  start: number[]
  path: Vec3[]
  duration: number
}

/** A held command: normalized base velocity and lift direction. */
export interface JogHold {
  drive: Vec3
  lift: -1 | 0 | 1
}

export const IDLE_HOLD: JogHold = { drive: [0, 0, 0], lift: 0 }
/** Hold commands go out this often while the page is connected. */
export const HOLD_PERIOD_MS = 50

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), init)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error ?? `Request failed (${res.status})`)
  return body as T
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

export const fetchJogState = () => request<JogState>("/api/jog/state")

export const planJogMove = (arm: JogArm, pose: JogPose) =>
  post<JogPlan>("/api/jog/plan", { arm, position: pose.position, quaternion: pose.quaternion })

export const executeJogPlan = (id: number) =>
  post<{ id: number; duration: number }>("/api/jog/execute", { id })

export const cancelJogMove = () => post<{ stopped: boolean }>("/api/jog/cancel")

/**
 * Drive pad geometry: a pointer at (dx, dy) pixels from the pad centre, on a
 * pad of radius `radius`, as a normalized base command. Screen up is forward
 * (+x) and screen left is +y; the vector is clipped to the unit circle and
 * scaled by `speed` (0 to 1). Inside the centre `dead` fraction it is zero.
 */
export function padCommand(dx: number, dy: number, radius: number, speed = 1, dead = 0.08): Vec3 {
  if (!(radius > 0)) return [0, 0, 0]
  let forward = -dy / radius
  let left = -dx / radius
  const length = Math.hypot(forward, left)
  if (length < dead) return [0, 0, 0]
  if (length > 1) {
    forward /= length
    left /= length
  }
  const scale = Math.max(0, Math.min(1, speed))
  return [forward * scale, left * scale, 0]
}

/** Combine the drive pad and the turn buttons into one held command. */
export function combineHold(pad: Vec3, turn: -1 | 0 | 1, lift: -1 | 0 | 1, speed = 1): JogHold {
  const scale = Math.max(0, Math.min(1, speed))
  return { drive: [pad[0], pad[1], turn * scale], lift }
}

/** Gripper-tip error in millimetres and degrees, for the plan readout. */
export function describePlanError(plan: Pick<JogPlan, "positionError" | "orientationError">) {
  return {
    mm: plan.positionError * 1000,
    degrees: (plan.orientationError * 180) / Math.PI,
  }
}

/** The lift joint value in the URDF for a raised fraction (0 = fully raised). */
export function liftJointValue(raised: number | null | undefined, travel: number): number {
  if (raised == null || !Number.isFinite(raised)) return 0
  return (1 - Math.max(0, Math.min(1, raised))) * travel
}

const JOINT_SUFFIXES = ["s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0"]

/** Joint values the URDF should show for a jog state (and an optional plan). */
export function jointValuesFor(
  state: JogState | null,
  liftTravel: number,
  plan?: Pick<JogPlan, "arm" | "joints"> | null
): Record<string, number> {
  const values: Record<string, number> = {}
  for (const side of ["left", "right"] as const) {
    const joints = state?.arms?.[side]
    if (!joints) continue
    JOINT_SUFFIXES.forEach((suffix, i) => {
      values[`${side}_${suffix}`] = joints[i] ?? 0
    })
  }
  if (plan) {
    JOINT_SUFFIXES.forEach((suffix, i) => {
      values[`${plan.arm}_${suffix}`] = plan.joints[i] ?? 0
    })
  }
  const lift = liftJointValue(state?.lift?.raised, liftTravel)
  values.lift = lift
  values.lift_2 = lift
  if (state?.base) {
    values.base_x = state.base.x
    values.base_y = state.base.y
    values.base_yaw = state.base.yaw
  }
  return values
}

/**
 * Live link to the jog session.
 *
 * While `enabled`, opens `/api/jog/ws`, sends `holdRef.current` every
 * {@link HOLD_PERIOD_MS} (zeros included, so a release ramps down like the
 * real controllers) and returns the latest state the server pushes. Closing
 * the socket, hiding the tab or losing focus stops the base and lift: the
 * server treats a silent page as a dead command source.
 */
export function useJogLink(enabled: boolean, holdRef: { current: JogHold }) {
  const [state, setState] = useState<JogState | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return
    let closed = false
    let retry: ReturnType<typeof setTimeout> | undefined
    let timer: ReturnType<typeof setInterval> | undefined

    const connect = () => {
      const ws = new WebSocket(`${wsBaseUrl()}/api/jog/ws`)
      socketRef.current = ws
      ws.onopen = () => {
        setConnected(true)
        timer = setInterval(() => {
          if (ws.readyState !== WebSocket.OPEN) return
          const hold = document.hidden ? IDLE_HOLD : holdRef.current
          ws.send(JSON.stringify({ type: "hold", drive: hold.drive, lift: hold.lift }))
        }, HOLD_PERIOD_MS)
      }
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.type === "state") setState(message as JogState)
        else if (message.type === "error") setError(message.error)
      }
      ws.onclose = () => {
        clearInterval(timer)
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 1000)
      }
    }
    connect()
    return () => {
      closed = true
      clearTimeout(retry)
      clearInterval(timer)
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [enabled, holdRef])

  return { state, connected: enabled && connected, error }
}
