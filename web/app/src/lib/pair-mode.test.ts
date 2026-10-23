import type { AxolSettings } from "@almond/axol-vr-client"
import { describe, expect, it } from "vitest"

import { pairHints, pairKey, pairKind, pairLabel, pairModeOn } from "./pair-mode"

const settings = (values: AxolSettings["values"]): AxolSettings => ({ schema: [], values })

describe("pair mode", () => {
  it("is box mode with the parcel gripper's key", () => {
    const s = settings({ box_mode: true, reengage: "clutch" })
    expect(pairKind(s)).toBe("box")
    expect(pairKey("box")).toBe("box_mode")
    expect(pairModeOn(s)).toBe(true)
    expect(pairLabel("box")).toBe("Box")
  })

  it("is rigid mode with the stock gripper's key", () => {
    const s = settings({ rigid_mode: false, box_elbow_out: 30, reengage: "clutch" })
    expect(pairKind(s)).toBe("rigid")
    expect(pairKey("rigid")).toBe("rigid_mode")
    expect(pairModeOn(s)).toBe(false)
    expect(pairModeOn(settings({ rigid_mode: true }))).toBe(true)
    expect(pairLabel("rigid")).toBe("Rigid")
  })

  it("is absent before the server announces one", () => {
    expect(pairKind(null)).toBeNull()
    expect(pairKind(settings({ reengage: "clutch" }))).toBeNull()
    expect(pairModeOn(null)).toBe(false)
  })

  it("has no width or grasp controls in rigid mode", () => {
    const rows = pairHints("rigid").map(([, action]) => action)
    expect(rows.join(" ")).not.toMatch(/width|grasp/)
    expect(
      pairHints("box")
        .map(([, action]) => action)
        .join(" ")
    ).toMatch(/width/)
  })
})
