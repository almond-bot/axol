import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"

import { parseKinematics, posesFromJoints } from "./episode-fk"

// Tests run from web/ (vitest.config.ts).
const URDF = resolve(process.cwd(), "../almond_axol/kinematics/urdf/axol.urdf")
const NAMES = ["shoulder_1", "shoulder_2", "shoulder_3", "elbow", "wrist_1", "wrist_2", "wrist_3"]

function joints(left: number[], right: number[]) {
  const out: Record<string, (number | null)[]> = {}
  NAMES.forEach((name, i) => {
    out[`left_${name}.pos`] = [left[i]]
    out[`right_${name}.pos`] = [right[i]]
  })
  return out
}

// Reference poses from the Host Computer's MuJoCo forward kinematics
// (almond_axol.kinematics.mujoco_fk.AxolForwardKinematics) on axol.urdf.
const Q = [0.3, -0.4, 0.5, 1.1, -0.2, 0.6, 0.9]
const REFERENCE = {
  zero: {
    left: [0, 0.19958, 0.14805, 0, 0, 1.5708],
    right: [0, -0.19958, 0.14805, 0, 0, 1.57079],
  },
  bent: {
    left: [0.34726, 0.55082, 0.57767, -0.34498, -2.68298, 0.54244],
    right: [0.34726, -0.55082, 0.57767, -2.67884, -0.34445, 0.53359],
  },
}

function expectPose(pose: Record<string, (number | null)[]> | undefined, expected: number[]) {
  const got = ["x", "y", "z", "rx", "ry", "rz"].map((axis) => pose?.[axis][0])
  got.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 3))
}

describe("posesFromJoints", () => {
  const robot = parseKinematics(readFileSync(URDF, "utf8"))

  it("matches the Host Computer's forward kinematics", () => {
    const zero = posesFromJoints(robot, joints(Array(7).fill(0), Array(7).fill(0)), 1)
    expectPose(zero.left, REFERENCE.zero.left)
    expectPose(zero.right, REFERENCE.zero.right)
    const bent = posesFromJoints(
      robot,
      joints(
        Q,
        Q.map((v) => -v)
      ),
      1
    )
    expectPose(bent.left, REFERENCE.bent.left)
    expectPose(bent.right, REFERENCE.bent.right)
  })

  it("takes recorded angles past a joint's limit as they are", () => {
    // Shoulder 2 is 0.546 rad, past its 0.349 rad upper limit.
    const past = [-0.4568, 0.5463, 0, 0.7027, 0, 0.5257, 0]
    const { left } = posesFromJoints(robot, joints(past, Array(7).fill(0)), 1)
    // Reference: the Host Computer's MuJoCo forward kinematics, which doesn't clamp.
    expect(left?.x[0]).toBeCloseTo(-0.0484, 3)
    expect(left?.y[0]).toBeCloseTo(-0.0843, 3)
    expect(left?.z[0]).toBeCloseTo(0.2675, 3)
  })

  it("leaves out a side without joints and gives nulls for a missing angle", () => {
    const only = joints(Q, Q)
    for (const name of NAMES) delete only[`right_${name}.pos`]
    only["left_elbow.pos"] = [null]
    const poses = posesFromJoints(robot, only, 1)
    expect(poses.right).toBeUndefined()
    expect(poses.left?.x).toEqual([null])
  })
})
