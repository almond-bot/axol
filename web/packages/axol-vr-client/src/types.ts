import type { QuaternionLike, Vector3Like } from "three"

export enum AxolState {
  Teleop = "teleop",
  DataCollection = "data_collection",
  Recording = "recording",
  Saving = "saving",
  Error = "error",
}

/**
 * Operating mode the server locks the headset HUD to, pushed once on connect as
 * `{"type":"mode","value":...}`. "teleop" (from `axol teleop`) hides the
 * data-collection + recording controls; "data_collection" (from `axol
 * collect-data`) allows recording but not switching back to plain teleop.
 */
export type AxolMode = "teleop" | "data_collection"

/**
 * Which episode action a HUD confirmation popup is gating while recording:
 * stopping to save the episode ("save", armed by A) or discarding it to
 * re-record ("discard", armed by X). Null when no confirmation is pending.
 */
export type ConfirmAction = "save" | "discard"

export enum AxolConnectionStatus {
  Idle = "idle",
  Connecting = "connecting",
  Open = "open",
  Error = "error",
  Failed = "failed",
}

export type AxolPoseData = {
  l_ee: { position: Vector3Like; quaternion: QuaternionLike }
  r_ee: { position: Vector3Like; quaternion: QuaternionLike }
  l_elbow: Vector3Like
  r_elbow: Vector3Like
  l_lock: boolean
  r_lock: boolean
  l_grip: number
  r_grip: number
  reset: boolean
  state: AxolState
  /** Monotonic per-connection frame counter. */
  seq?: number
  /** Capture timestamp (ms, `performance.now()`) for server-side interpolation. */
  t?: number
  /** Left thumbstick x, [-1, 1], right = +1 — Jelly strafe. */
  l_stick_x?: number
  /** Left thumbstick y, [-1, 1], pushed forward = -1 — Jelly drive. */
  l_stick_y?: number
  /** Right thumbstick x, [-1, 1], right = +1 — Jelly rotation. */
  r_stick_x?: number
  /** Right thumbstick y, [-1, 1], pushed forward = -1 — box-mode jog only. */
  r_stick_y?: number
  /** Left thumbstick pressed in — lift down while held. */
  l_stick_click?: boolean
  /**
   * Right thumbstick pressed in — lift up while held (in box mode: a jog
   * modifier — leader stick up/yaw, other stick fingertip tilt). Both sticks
   * clicked together toggle box mode (the headset sends the `set` message
   * itself, see `AxolVRClient.onBothStickClick`).
   */
  r_stick_click?: boolean
}

/** Re-engage behaviour (the `reengage` live setting): "clutch" — a re-engaging
 * arm stays put and the controller's current pose becomes its origin (you match
 * the arm); "ramp" — the arm eases out to where the controller is under the
 * mapping from its previous engage (the arm matches you). */
export type AxolReengage = "clutch" | "ramp"

/**
 * One live session setting as published by the server (see `AxolSettings`).
 * The schema is rendered generically by the HUD and the control panel: a
 * boolean is a toggle, a select cycles/lists its `options`, a number steps
 * between `min` and `max` by `step`.
 */
export type AxolSettingDef = {
  key: string
  label: string
  type: "boolean" | "select" | "number"
  help: string
  options: string[]
  min: number | null
  max: number | null
  step: number | null
  unit: string
}

/**
 * The server's live session settings, pushed as
 * `{"type":"settings","value":AxolSettings}` on connect and after every change
 * (from any client). Change one with `{"type":"set","key","value"}` — see
 * `useAxolSettings`; a boolean key also takes the value `"toggle"`, flipped
 * server-side against the value the server holds. The value set includes at
 * least `box_mode` (boolean) and `reengage` (`AxolReengage`).
 */
export type AxolSettings = {
  schema: AxolSettingDef[]
  values: Record<string, boolean | number | string>
}

/**
 * Live joint state the teleop server pushes ~20x/s as
 * `{"type":"joints","value":AxolJointState}` — drives the in-headset ghost
 * robot. `q` maps URDF joint names (e.g. `left_s1_0`) to radians; grips are
 * normalised 0 (closed) – 1 (open). `pair` is the gripper-pair geometry from
 * the IK worker: `aligned` when the grippers already form the box-mode pair
 * (fingers forward, a flat face toward each other across a box-mode-sized gap
 * — a good moment to switch to box mode), `width` in metres, `tilt` the pair's
 * inward fingertip yaw in degrees (jogged live in box mode). Null until the
 * worker's first report.
 */
export type AxolJointState = {
  q: Record<string, number>
  l_grip: number
  r_grip: number
  engaged: boolean
  pair: { aligned: boolean; width: number; tilt: number } | null
}
