import type { AxolMode, AxolPoseSourceKind } from "./types"

/** Whether this WebXR client is the session's permitted pose/controller class. */
export function webxrCanControlPose(expectedPoseSourceKind: AxolPoseSourceKind): boolean {
  return expectedPoseSourceKind !== "tracker"
}

/** Whether this WebXR client may originate data-collection controls. */
export function webxrCanControlRecording(
  mode: AxolMode | null,
  expectedPoseSourceKind: AxolPoseSourceKind
): boolean {
  return mode !== "teleop" && webxrCanControlPose(expectedPoseSourceKind)
}
