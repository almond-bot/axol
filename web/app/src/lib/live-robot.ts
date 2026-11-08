/**
 * Joint values for the headset's live robot model (see
 * `components/live-robot.tsx`), from the teleop server's `joints` push.
 * Kept free of three.js so it tests without a renderer.
 */

/** Height (m) of the robot model at its default size, floor to top. */
export const LIVE_ROBOT_HEIGHT = 0.45

// Joints the model reads straight from the push (radians), per arm.
const ARM_JOINTS = ["s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0"]

/** The parts of a joint push the model reads. */
export interface LiveRobotSample {
  q: Record<string, number>
  l_grip: number
  r_grip: number
  lift: number | null
}

/** The joints a loaded URDF has, with their travel. */
export interface LiveRobotJoints {
  joints: Record<string, { limit?: { upper?: number | string } } | undefined>
}

/**
 * Target URDF joint values for a joint push.
 *
 * Arms come through as is. Each gripper's 0 (closed) – 1 (open) drives its
 * first finger joint across its travel (the second finger mimics it in the
 * URDF). Jelly's lift height (1 = fully raised, null without a lift) drives
 * the `lift` joint, which counts down from fully raised (`lift_2` mimics it).
 * Joints the model doesn't have are skipped.
 */
export function liveRobotTargets(
  sample: LiveRobotSample,
  robot: LiveRobotJoints
): Record<string, number> {
  const out: Record<string, number> = {}
  for (const side of ["left", "right"] as const) {
    for (const suffix of ARM_JOINTS) {
      const name = `${side}_${suffix}`
      const value = sample.q[name]
      if (robot.joints[name] && Number.isFinite(value)) out[name] = value
    }
    const finger = robot.joints[`${side}_finger_1_joint`]
    const grip = side === "left" ? sample.l_grip : sample.r_grip
    const travel = Number(finger?.limit?.upper ?? 0)
    if (finger && Number.isFinite(grip) && travel > 0) {
      out[`${side}_finger_1_joint`] = Math.max(0, Math.min(1, grip)) * travel
    }
  }
  const lift = robot.joints.lift
  const travel = Number(lift?.limit?.upper ?? 0)
  if (lift && travel > 0) {
    const raised = sample.lift == null ? 1 : Math.max(0, Math.min(1, sample.lift))
    out.lift = (1 - raised) * travel
  }
  return out
}
