import { describe, expect, it } from "vitest"

import { DAMIAO_JOINTS, jointVendor, shownForJoint } from "./firmware-loop"
import { MYACTUATOR_JOINTS } from "./wire-mode"

describe("firmware loop vendors", () => {
  it("maps every arm joint to its motor vendor", () => {
    for (const j of MYACTUATOR_JOINTS) expect(jointVendor(j)).toBe("myactuator")
    for (const j of DAMIAO_JOINTS) expect(jointVendor(j)).toBe("damiao")
    expect(jointVendor("")).toBeNull()
    expect(jointVendor(undefined)).toBeNull()
    expect(jointVendor("gripper")).toBeNull()
  })

  it("hides a knob the selected joint's loop does not have", () => {
    // position_kd / current_* / planner accel exist only on MyActuator.
    expect(shownForJoint(["myactuator"], "elbow")).toBe(true)
    expect(shownForJoint(["myactuator"], "wrist_2")).toBe(false)
    // The ACC/DEC profiler exists only on the Damiao wrists.
    expect(shownForJoint(["damiao"], "wrist_3")).toBe(true)
    expect(shownForJoint(["damiao"], "shoulder_1")).toBe(false)
  })

  it("shows shared knobs always, and everything before a joint is picked", () => {
    expect(shownForJoint(undefined, "wrist_2")).toBe(true)
    expect(shownForJoint(["myactuator", "damiao"], "wrist_2")).toBe(true)
    expect(shownForJoint(["myactuator"], "")).toBe(true)
    expect(shownForJoint(["damiao"], undefined)).toBe(true)
  })
})
