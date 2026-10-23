import { describe, expect, it } from "vitest"

import { rebaseValues } from "./settings-rebase"

describe("rebaseValues", () => {
  it("lets untouched keys follow the host and keeps staged edits", () => {
    const seed = { "axol.left_stiffness": 0.8, "teleop.frequency": 240 }
    // The operator staged a stiffness change and reset the frequency.
    const draft = { "axol.left_stiffness": 0.5 }
    // Meanwhile the host saved a robot model and a new frequency.
    const next = {
      "axol.left_stiffness": 0.8,
      "teleop.frequency": 120,
      "kinematics.mount": "owl",
    }
    expect(rebaseValues(draft, seed, next)).toEqual({
      "axol.left_stiffness": 0.5,
      "kinematics.mount": "owl",
    })
  })

  it("drops untouched keys the host no longer stores", () => {
    expect(rebaseValues({ "a.b": 1, "c.d": 2 }, { "a.b": 1, "c.d": 2 }, { "c.d": 2 })).toEqual({
      "c.d": 2,
    })
  })
})
