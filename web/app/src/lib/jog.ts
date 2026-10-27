/**
 * Client for the jog session (`axol serve`'s `jog` operation).
 *
 * The page plans and executes arm moves over REST. Over `/api/jog/ws` it
 * streams hold commands (base velocity, lift direction, gripper jogs), asks
 * for IK previews of a dragged target, and receives the live session state.
 * Poses are in Axol's torso frame: +x forward, +y left, +z up, metres, at
 * the gripper tips.
 */
import { useCallback, useEffect, useRef, useState } from "react"
import type { MarionetteState } from "@/lib/marionette"
import {
  apiUrl,
  parseAxolModel,
  parseGripper,
  wsBaseUrl,
  type AxolModel,
  type GripperType,
} from "@/lib/supervisor"

export type JogArm = "left" | "right"
/** Axes a gripper jog moves along: the torso's, or the gripper's own. */
export type JogFrame = "torso" | "gripper"
export type Vec3 = [number, number, number]
/** A gripper jog: x, y, z, roll, pitch, yaw, each -1 to 1. */
export type JogTwist = [number, number, number, number, number, number]
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
  hold?: { active: boolean; drive: Vec3; lift: number; arms?: Record<JogArm, number[]> }
  /** Per arm: jogging, and held back by a joint limit or the body. */
  armJog?: Record<JogArm, { moving: boolean; limited: boolean }> | null
  execution?: { id: number; arm: JogArm; progress: number } | null
  lastExecution?: { id: number; arm: JogArm; status: "done" | "stopped" } | null
  /** Phones, arms, Stop and cameras, while the Marionette operation runs. */
  marionette?: MarionetteState
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

/** The closest reachable pose for a target, solved live while it is dragged. */
export interface JogPreview {
  seq: number
  arm: JogArm
  target: JogPose
  reached: JogPose
  reachable: boolean
  positionError: number
  orientationError: number
  joints: number[]
}

/** A held command: normalized base velocity, lift direction and gripper jogs. */
export interface JogHold {
  drive: Vec3
  lift: -1 | 0 | 1
  arms?: Partial<Record<JogArm, JogTwist>>
  armFrame?: JogFrame
  /** Gripper jog speed, 0 to 1. */
  armSpeed?: number
}

export const IDLE_HOLD: JogHold = { drive: [0, 0, 0], lift: 0 }
export const NO_TWIST: JogTwist = [0, 0, 0, 0, 0, 0]
/** Hold commands go out this often while the page is connected. */
export const HOLD_PERIOD_MS = 50
/** A preview that hasn't come back in this long is given up on. */
export const PREVIEW_TIMEOUT_MS = 1000

const SIM_ROBOT_KEY = "axolSimRobot"

/** The sim robot this browser last picked, if it is still a valid one. */
export function storedSimRobot(): { model: AxolModel; gripper: GripperType } | null {
  try {
    const raw = JSON.parse(localStorage.getItem(SIM_ROBOT_KEY) ?? "null")
    const model = parseAxolModel(raw?.model)
    return model ? { model, gripper: parseGripper(raw?.gripper) } : null
  } catch {
    return null
  }
}

/** Remember the sim robot picked for the next visit. */
export function storeSimRobot(robot: { model: AxolModel; gripper: GripperType }) {
  localStorage.setItem(SIM_ROBOT_KEY, JSON.stringify(robot))
}

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

/** Add one arm's gripper jog to a held command. */
export function withArmJog(
  hold: JogHold,
  arm: JogArm,
  twist: readonly number[],
  frame: JogFrame,
  speed: number
): JogHold {
  const clip = (v: number) => Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0))
  return {
    ...hold,
    arms: { [arm]: NO_TWIST.map((_, i) => clip(twist[i] ?? 0)) as JogTwist },
    armFrame: frame,
    armSpeed: Math.max(0, clip(speed)),
  }
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

/** Joint values the URDF should show for a jog state (and an optional plan
 *  or preview, for one arm). */
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
 * Preview requests, latest first: at most one is on the wire, and while it
 * is, newer targets replace each other until its answer comes back (or it
 * times out). Sequence numbers only grow, so the server can drop anything
 * older than what it has seen and the page never shows an older answer
 * over a newer one.
 */
export class PreviewQueue {
  private seq = 0
  private inFlight = 0
  private sentAt = 0
  private shown = 0
  private pending: { seq: number; arm: JogArm; pose: JogPose } | null = null

  private readonly timeoutMs: number

  constructor(timeoutMs = PREVIEW_TIMEOUT_MS) {
    this.timeoutMs = timeoutMs
  }

  /** Queue a target, replacing any not yet sent. */
  request(arm: JogArm, pose: JogPose) {
    this.seq += 1
    this.pending = { seq: this.seq, arm, pose }
  }

  /** The preview message to send at time `now` (ms), if one is due. */
  next(now: number) {
    if (!this.pending) return null
    if (this.inFlight && now - this.sentAt < this.timeoutMs) return null
    const { seq, arm, pose } = this.pending
    this.pending = null
    this.inFlight = seq
    this.sentAt = now
    return { type: "preview", seq, arm, position: pose.position, quaternion: pose.quaternion }
  }

  /** An answer arrived; true if it is newer than any shown, so show it. */
  answer(seq: number, failed = false) {
    if (seq === this.inFlight) this.inFlight = 0
    if (failed || seq <= this.shown) return false
    this.shown = seq
    return true
  }

  /** The link dropped: nothing is on the wire any more. */
  reconnected() {
    this.inFlight = 0
  }
}

/** The hold message for a held command. */
export function holdMessage(hold: JogHold) {
  return {
    type: "hold",
    drive: hold.drive,
    lift: hold.lift,
    ...(hold.arms ? { arms: hold.arms } : {}),
    ...(hold.armFrame ? { armFrame: hold.armFrame } : {}),
    ...(hold.armSpeed != null ? { armSpeed: hold.armSpeed } : {}),
  }
}

/**
 * Live link to the jog session.
 *
 * While `enabled`, opens `/api/jog/ws`, sends `holdRef.current` every
 * {@link HOLD_PERIOD_MS} (zeros included, so a release ramps down like the
 * real controllers) and returns the latest state the server pushes. Closing
 * the socket, hiding the tab or losing focus stops the base, lift and arms:
 * the server treats a silent page as a dead command source.
 *
 * `requestPreview` asks for the IK preview of a target. At most one request
 * is on the wire; while it is, newer targets replace each other and the
 * latest goes out when the answer comes back, so previews keep up with
 * whatever the host can solve. `preview` is the newest answer.
 */
export function useJogLink(enabled: boolean, holdRef: { current: JogHold }) {
  const [state, setState] = useState<JogState | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<JogPreview | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const queueRef = useRef<PreviewQueue | null>(null)
  const queue = () => (queueRef.current ??= new PreviewQueue())

  const flushPreview = useCallback(() => {
    const ws = socketRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const message = (queueRef.current ??= new PreviewQueue()).next(performance.now())
    if (message) ws.send(JSON.stringify(message))
  }, [])

  const requestPreview = useCallback(
    (arm: JogArm, pose: JogPose) => {
      ;(queueRef.current ??= new PreviewQueue()).request(arm, pose)
      flushPreview()
    },
    [flushPreview]
  )

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
        queue().reconnected()
        flushPreview()
        timer = setInterval(() => {
          if (ws.readyState !== WebSocket.OPEN) return
          const hold = document.hidden ? IDLE_HOLD : holdRef.current
          ws.send(JSON.stringify(holdMessage(hold)))
          // Recovers a preview whose answer was lost.
          flushPreview()
        }, HOLD_PERIOD_MS)
      }
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.type === "state") setState(message as JogState)
        else if (message.type === "error") setError(message.error)
        else if (message.type === "preview") {
          if (queue().answer(message.seq, Boolean(message.error))) setPreview(message as JogPreview)
          flushPreview()
        }
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
  }, [enabled, holdRef, flushPreview])

  return { state, connected: enabled && connected, error, preview, requestPreview }
}
