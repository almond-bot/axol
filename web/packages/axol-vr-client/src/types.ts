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
}
