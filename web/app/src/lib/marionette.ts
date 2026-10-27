/**
 * Client for the Marionette operation (`axol serve`'s `marionette` command):
 * iPhones running Marionette drive Axol's arms through a gateway, and the
 * desktop page watches the phones, shows the cameras and drives Jelly and the
 * lift from the keyboard.
 *
 * While the operation runs, `/api/jog/ws` streams the jog state with a
 * `marionette` key ({@link MarionetteState}) and takes the page's base and
 * lift holds; `/api/marionette/stop` sets the desktop Stop,
 * `/api/marionette/reset` sends an arm back to rest and
 * `/api/marionette/cameras` streams JPEG frames.
 */
import { combineHold, padCommand, type JogArm, type JogHold, type Vec3 } from "@/lib/jog"
import { apiUrl } from "@/lib/supervisor"

export type MarionetteTracking =
  | "normal"
  | "limited"
  | "initializing"
  | "relocalizing"
  | "unavailable"
export type MarionettePoseSource = "arkit" | "device-motion" | "synthetic"
export type MarionetteCalibration = "uncalibrated" | "capturing" | "captured"

export interface MarionettePhone {
  id: string
  name: string
  /** CSS colour the phone shows itself in. */
  color: string
  connected: boolean
  tracking: MarionetteTracking
  source: MarionettePoseSource | null
  calibration: MarionetteCalibration
  arm: JogArm
  /** The camera view picked on the phone. */
  view: string
  /** Motion scale, phone to gripper. */
  scale: number
  /** Drive speed, 0 to 1. */
  speed: number
  /** Hold to move is pressed. */
  deadman: boolean
  /** This phone's Stop is on. */
  stop: boolean
  /** Its arm is following it. */
  engaged: boolean
  /** It is driving the base or the lift. */
  driving: boolean
  /** Seconds since its last pose, null before the first. */
  poseAge: number | null
}

export interface MarionetteArm {
  /** Id of the phone that has the arm. */
  owner: string | null
  ownerName: string | null
  engaged: boolean
  /** Held back by its reach, a joint limit or the body. */
  limited: boolean
  /** Commanded closure: 0 open, 1 closed. */
  grip: number
  /** Why the arm won't move, as a sentence for the operator. */
  reason: string | null
  /** On its way back to the rest pose. */
  resetting: boolean
}

export interface MarionetteState {
  gateway: { address: string; connected: boolean; error: string | null }
  phones: MarionettePhone[]
  arms: Record<JogArm, MarionetteArm>
  /** `by` names who turned it on ("Phone 2", "Desktop"); `desktop` is this page's Stop. */
  stop: { on: boolean; by: string | null; desktop: boolean }
  cameras: {
    available: string[]
    source: "sim" | "zed" | "synthetic" | null
    error: string | null
  }
}

/** Where Axol reaches the gateway when nothing else is entered. */
export const DEFAULT_GATEWAY = "localhost:8787"
export const GATEWAY_STORAGE_KEY = "axolMarionetteGateway"
export const CAMERA_STORAGE_KEY = "axolMarionetteCameras"
/** Camera frames per second the page asks for. */
export const CAMERA_FPS = 10

// -- Cameras ---------------------------------------------------------------

export type CameraStream = "head" | "left" | "right"

export const CAMERA_STREAMS: readonly { id: CameraStream; label: string }[] = [
  { id: "head", label: "Head" },
  { id: "left", label: "Left wrist" },
  { id: "right", label: "Right wrist" },
]

export function isCameraStream(value: unknown): value is CameraStream {
  return CAMERA_STREAMS.some((stream) => stream.id === value)
}

/** "Left wrist", for a pane's label. */
export function cameraLabel(stream: CameraStream): string {
  return CAMERA_STREAMS.find((s) => s.id === stream)?.label ?? stream
}

/** "left wrist camera", for a sentence. */
export function cameraNoun(stream: CameraStream): string {
  return `${cameraLabel(stream).toLowerCase()} camera`
}

/** The panes to show from what the page stored: every camera unless a valid choice was kept. */
export function parseCameraChoice(raw: string | null): CameraStream[] {
  if (raw == null) return CAMERA_STREAMS.map((s) => s.id)
  try {
    const value: unknown = JSON.parse(raw)
    if (!Array.isArray(value)) return CAMERA_STREAMS.map((s) => s.id)
    return CAMERA_STREAMS.map((s) => s.id).filter((id) => value.includes(id))
  } catch {
    return CAMERA_STREAMS.map((s) => s.id)
  }
}

/** Path of the camera socket for some streams, in a fixed order. */
export function cameraSocketPath(streams: readonly CameraStream[], fps = CAMERA_FPS): string {
  const ids = CAMERA_STREAMS.map((s) => s.id).filter((id) => streams.includes(id))
  return `/api/marionette/cameras?streams=${ids.join(",")}&fps=${fps}`
}

/**
 * A binary camera message: the stream id in UTF-8, a newline byte, then the
 * JPEG. Null when there is no newline, no id or no image.
 */
export function parseCameraFrame(
  data: ArrayBuffer
): { stream: string; jpeg: Uint8Array<ArrayBuffer> } | null {
  const bytes = new Uint8Array(data)
  const split = bytes.indexOf(0x0a)
  if (split <= 0 || split === bytes.length - 1) return null
  return {
    stream: new TextDecoder().decode(bytes.subarray(0, split)),
    jpeg: bytes.subarray(split + 1),
  }
}

export interface CameraInfo {
  available: CameraStream[]
  error: string | null
}

/** A text camera message, `{"type": "cameras", available, error}`; null for anything else. */
export function parseCameraInfo(text: string): CameraInfo | null {
  try {
    const message = JSON.parse(text)
    if (message?.type !== "cameras") return null
    return {
      available: Array.isArray(message.available) ? message.available.filter(isCameraStream) : [],
      error: typeof message.error === "string" && message.error ? message.error : null,
    }
  } catch {
    return null
  }
}

// -- Keys ------------------------------------------------------------------

export type DriveKey = "w" | "a" | "s" | "d" | "q" | "e" | "z" | "x"

const KEY_CODES: Record<string, DriveKey> = {
  KeyW: "w",
  KeyA: "a",
  KeyS: "s",
  KeyD: "d",
  KeyQ: "q",
  KeyE: "e",
  KeyZ: "z",
  KeyX: "x",
}
const DRIVE_KEYS = new Set<string>(Object.values(KEY_CODES))

/** The drive key an event is for, by its place on the keyboard; null for any other key. */
export function driveKeyOf(event: { code?: string; key?: string }): DriveKey | null {
  if (event.code) return KEY_CODES[event.code] ?? null
  const key = event.key?.toLowerCase() ?? ""
  return DRIVE_KEYS.has(key) ? (key as DriveKey) : null
}

const axis = (held: ReadonlySet<DriveKey>, plus: DriveKey, minus: DriveKey) =>
  ((held.has(plus) ? 1 : 0) - (held.has(minus) ? 1 : 0)) as -1 | 0 | 1

/**
 * Held keys as axes: W/S drive forward and back, A/D left and right, Q/E
 * turn left and right, Z/X raise and lower the lift. Opposing keys cancel.
 */
export function keyAxes(keys: Iterable<DriveKey>): { drive: Vec3; lift: -1 | 0 | 1 } {
  const held = new Set(keys)
  return {
    drive: [axis(held, "w", "s"), axis(held, "a", "d"), axis(held, "q", "e")],
    lift: axis(held, "z", "x"),
  }
}

// -0 reads as 0 in a hold message and in a test.
const tidy = (v: number) => (v === 0 ? 0 : v)

/**
 * The page's held command: the keys merged with the on-screen drive pad,
 * turn buttons and lift buttons.
 *
 * Keys drive like the pad pushed to its rim, so a diagonal is no faster than
 * straight ahead, and both are scaled by `speed` the way the pad is. `pad` is
 * the pad's command, already scaled. Keys and pad add up, capped at `speed`.
 * Turning scales by `speed` like the turn buttons; the lift has no speed.
 */
export function desktopHold(
  keys: Iterable<DriveKey>,
  controls: { pad: Vec3; turn: -1 | 0 | 1; lift: -1 | 0 | 1 },
  speed: number
): JogHold {
  const { drive, lift } = keyAxes(keys)
  const scale = Math.max(0, Math.min(1, speed))
  const keyed = padCommand(-drive[1], -drive[0], 1, scale)
  let x = keyed[0] + controls.pad[0]
  let y = keyed[1] + controls.pad[1]
  const length = Math.hypot(x, y)
  if (length > scale) {
    x = (x * scale) / length
    y = (y * scale) / length
  }
  const turn = Math.sign(drive[2] + controls.turn) as -1 | 0 | 1
  const hold = combineHold([x, y, 0], turn, Math.sign(lift + controls.lift) as -1 | 0 | 1, scale)
  return { ...hold, drive: hold.drive.map(tidy) as Vec3 }
}

const NON_TEXT_INPUTS = new Set([
  "button",
  "checkbox",
  "color",
  "file",
  "image",
  "radio",
  "range",
  "reset",
  "submit",
])

/** True while focus is somewhere keys type text or pick an option, so drive keys stay out of it. */
export function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tag = target.tagName
  if (tag === "TEXTAREA" || tag === "SELECT") return true
  return tag === "INPUT" && !NON_TEXT_INPUTS.has((target as HTMLInputElement).type)
}

// -- Operator text ---------------------------------------------------------

export const TRACKING_LABELS: Record<MarionetteTracking, string> = {
  normal: "Tracking",
  limited: "Tracking limited",
  initializing: "Tracking starting",
  relocalizing: "Tracking recovering",
  unavailable: "No tracking",
}

export const CALIBRATION_LABELS: Record<MarionetteCalibration, string> = {
  uncalibrated: "Not calibrated",
  capturing: "Calibrating",
  captured: "Calibrated",
}

/** One line on what a phone is doing: "Moving the left arm". */
export function phoneActivity(
  phone: MarionettePhone,
  arms?: Partial<Record<JogArm, MarionetteArm>> | null
): string {
  if (!phone.connected) return "Offline"
  if (phone.stop) return "Stop is on"
  if (phone.engaged) return `Moving the ${phone.arm} arm`
  if (phone.driving) return "Driving the base"
  if (phone.calibration === "capturing") return "Calibrating"
  const owner = arms?.[phone.arm]
  if (owner?.owner && owner.owner !== phone.id)
    return `${owner.ownerName ?? "Another phone"} has the ${phone.arm} arm`
  return "Idle"
}

const isLoopback = (host: string) => host === "localhost" || host === "::1" || /^127\./.test(host)

/**
 * The sentence telling an operator where to point the Marionette app.
 * `axolHost` is the Axol computer's address as this page reaches it: a
 * gateway local to Axol is reached at that address.
 */
export function phoneConnectHint(address: string, axolHost = ""): string {
  const typed = address.trim() || DEFAULT_GATEWAY
  let host = ""
  let port = ""
  try {
    const url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(typed) ? typed : `ws://${typed}`)
    host = url.hostname.replace(/^\[|\]$/g, "")
    port = url.port
  } catch {
    return `In the Marionette app, connect to ${typed}.`
  }
  if (!isLoopback(host)) return `In the Marionette app, connect to ${typed}.`
  const axol = axolHost.replace(/^\[|\]$/g, "")
  if (axol && !isLoopback(axol)) {
    const named = axol.includes(":") ? `[${axol}]` : axol
    return `In the Marionette app, connect to ${port ? `${named}:${port}` : named}.`
  }
  return port
    ? `In the Marionette app, connect to port ${port} on the computer running the gateway.`
    : "In the Marionette app, connect to the computer running the gateway."
}

// -- API -------------------------------------------------------------------

/** Turn the desktop Stop on or off. */
export async function setMarionetteStop(on: boolean): Promise<{ on: boolean; by: string | null }> {
  const res = await fetch(apiUrl("/api/marionette/stop"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ on }),
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error ?? `Request failed (${res.status})`)
  return body
}

/** Send one arm back to its rest pose. */
export async function resetMarionetteArm(arm: JogArm): Promise<{ arm: JogArm }> {
  const res = await fetch(apiUrl("/api/marionette/reset"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arm }),
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.error ?? `Request failed (${res.status})`)
  return body
}
