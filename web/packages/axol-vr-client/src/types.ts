import type { QuaternionLike, Vector3Like } from "three"

export enum AxolState {
  Teleop = "teleop",
  DataCollection = "data_collection",
  Recording = "recording",
  Saving = "saving",
  Error = "error",
}

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
}
