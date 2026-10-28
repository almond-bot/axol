/** Pose arithmetic for the Jog page's nudge buttons. */
import { Quaternion, Vector3 } from "three"
import type { JogPose } from "@/lib/jog"

const AXES = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)] as const

/**
 * Nudge a target pose.
 *
 * `move` shifts it `amount` metres along a torso axis (x forward, y left,
 * z up). `turn` rotates it `amount` radians about one of the gripper's own
 * axes, keeping the position.
 */
export function nudgePose(
  pose: JogPose,
  kind: "move" | "turn",
  axis: 0 | 1 | 2,
  amount: number
): JogPose {
  if (kind === "move") {
    const position = [...pose.position] as JogPose["position"]
    position[axis] += amount
    return { position, quaternion: [...pose.quaternion] as JogPose["quaternion"] }
  }
  const current = new Quaternion(...pose.quaternion)
  const turn = new Quaternion().setFromAxisAngle(AXES[axis], amount)
  const next = current.multiply(turn).normalize()
  return {
    position: [...pose.position] as JogPose["position"],
    quaternion: [next.x, next.y, next.z, next.w],
  }
}
