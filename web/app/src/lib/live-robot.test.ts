import { describe, expect, it } from "vitest"
import { liveRobotTargets, type LiveRobotJoints } from "./live-robot"

const jelly: LiveRobotJoints = {
  joints: {
    left_s1_0: {},
    left_w2_0: {},
    right_e1_0: {},
    left_finger_1_joint: { limit: { upper: 0.0588 } },
    right_finger_1_joint: { limit: { upper: "0.0588" } },
    lift: { limit: { upper: 0.34 } },
  },
}

const sample = { q: {}, l_grip: 1, r_grip: 1, lift: null }

describe("liveRobotTargets", () => {
  it("passes the arm joints the model has", () => {
    const out = liveRobotTargets(
      { ...sample, q: { left_s1_0: 0.3, right_e1_0: -1.2, left_e2_0: 0.5 } },
      jelly
    )
    expect(out.left_s1_0).toBe(0.3)
    expect(out.right_e1_0).toBe(-1.2)
    expect(out).not.toHaveProperty("left_e2_0")
  })

  it("opens the fingers across their travel", () => {
    const out = liveRobotTargets({ ...sample, l_grip: 0.5, r_grip: 2 }, jelly)
    expect(out.left_finger_1_joint).toBeCloseTo(0.0294)
    expect(out.right_finger_1_joint).toBeCloseTo(0.0588)
  })

  it("lowers the lift from fully raised", () => {
    expect(liveRobotTargets({ ...sample, lift: 1 }, jelly).lift).toBe(0)
    expect(liveRobotTargets({ ...sample, lift: 0.25 }, jelly).lift).toBeCloseTo(0.255)
    expect(liveRobotTargets({ ...sample, lift: null }, jelly).lift).toBe(0)
  })

  it("skips the fingers and lift on a model without them", () => {
    const out = liveRobotTargets({ ...sample, lift: 0.5 }, { joints: { left_s1_0: {} } })
    expect(out).toEqual({})
  })
})
