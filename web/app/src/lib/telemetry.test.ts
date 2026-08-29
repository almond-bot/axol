import { describe, expect, it } from "vitest"

import { JOINTS, jointLabel, jointsFor, motorKey } from "./telemetry"

describe("telemetry labels", () => {
  it("drops only the gripper for a gripperless robot", () => {
    expect(jointsFor(false)).toHaveLength(JOINTS.length - 1)
    expect(jointsFor(false)).not.toContain("GRIPPER")
    expect(jointsFor(undefined)).toBe(JOINTS)
  })

  it("builds stable labels and keys", () => {
    expect(jointLabel("SHOULDER_1")).toBe("shoulder 1")
    expect(motorKey("left", "ELBOW")).toBe("left:ELBOW")
  })
})
