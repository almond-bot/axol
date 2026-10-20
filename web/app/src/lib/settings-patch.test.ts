import { describe, expect, it } from "vitest"

import { applySavedSettingValues, diffSettingValues } from "./settings-patch"

describe("settings draft diff", () => {
  it("preserves settings added after an edited draft was opened", () => {
    const base = { "axol.left.gripper.torque_limit": 0.5 }
    const draft = { "axol.left.gripper.torque_limit": 1 }
    // A fresh server snapshot may now also contain
    // mantis.left_trigger_full_scale=0.90. It was not in this draft's base,
    // so Save must change only the torque value.
    expect(diffSettingValues(base, draft, new Set())).toEqual({
      "axol.left.gripper.torque_limit": 1,
    })
  })

  it("still sends explicit resets and excludes independently managed keys", () => {
    const base = {
      "mantis.left_trigger_full_scale": 0.9,
      "system.hardware_profile": "mantis",
    }
    const draft = { "system.hardware_profile": "axol" }
    expect(diffSettingValues(base, draft, new Set(["system.hardware_profile"]))).toEqual({
      "mantis.left_trigger_full_scale": null,
    })
  })

  it("clears saved edits without hiding edits made while Save was running", () => {
    const base = { "axol.left.gripper.torque_limit": 0.5 }
    const saved = { "axol.left.gripper.torque_limit": 1 }
    const rebased = applySavedSettingValues(base, saved)
    expect(diffSettingValues(rebased, saved, new Set())).toEqual({})
    expect(
      diffSettingValues(rebased, { ...saved, "mantis.left_trigger_full_scale": 0.9 }, new Set())
    ).toEqual({ "mantis.left_trigger_full_scale": 0.9 })
  })
})
