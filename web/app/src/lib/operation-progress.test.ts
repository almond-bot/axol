import { describe, expect, it } from "vitest"

import { startupView } from "./operation-progress"
import type { OperationProgress } from "./supervisor"

const axol = { device: "Axol", sim: false }
const T = 1_000_000

const at = (
  steps: string[],
  current: number,
  startedAgo = 0,
  ready = false
): OperationProgress => ({
  steps,
  current,
  stepStartedAt: T - startedAgo,
  ready,
})

describe("startupView", () => {
  it("shows nothing once ready, or when the host reports no steps", () => {
    expect(startupView(at(["arms"], 0, 0, true), axol, T * 1000)).toBeNull()
    expect(startupView(at([], -1), axol, T * 1000)).toBeNull()
    expect(startupView(null, axol, T * 1000)).toBeNull()
  })

  it("names the step under way, with the device or the simulator", () => {
    expect(startupView(at(["headset", "arms", "ik"], 1), axol, T * 1000)?.label).toBe(
      "Connecting to Axol"
    )
    expect(startupView(at(["arms"], 0), { device: "Mantis", sim: false }, T * 1000)?.label).toBe(
      "Connecting to Mantis"
    )
    expect(startupView(at(["arms"], 0), { device: "Axol", sim: true }, T * 1000)?.label).toBe(
      "Starting the simulator"
    )
    expect(startupView(at(["ik"], 0), axol, T * 1000)?.label).toBe("Warming up IK")
    expect(startupView(at(["something-new"], 0), axol, T * 1000)?.label).toBe("Starting")
  })

  it("counts finished steps by their share and creeps within the current one", () => {
    const steps = ["headset", "arms", "ik"] // 1 + 4 + 15 seconds
    const start = startupView(at(steps, 2, 0), axol, T * 1000)!.percent
    expect(start).toBe(25)
    const later = startupView(at(steps, 2, 10), axol, T * 1000)!.percent
    expect(later).toBeGreaterThan(start)
    const stuck = startupView(at(steps, 2, 600), axol, T * 1000)!.percent
    expect(stuck).toBeLessThan(100)
    expect(stuck).toBeGreaterThan(later)
  })

  it("starts at zero before the first step", () => {
    expect(startupView(at(["arms"], -1), axol, T * 1000)).toEqual({
      label: "Starting",
      percent: 0,
    })
  })
})
