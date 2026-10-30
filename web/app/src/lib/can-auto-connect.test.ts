import { describe, expect, it } from "vitest"

import {
  chooseDetectedHardwareProfile,
  hardwarePresenceSignature,
  robotLinkReleasable,
  shouldStartCanDiscovery,
} from "./can-auto-connect"
import type { CanProfileInventory } from "./supervisor"

function inventory(
  axol: boolean,
  mantis: boolean,
  suppressed: Partial<Record<"axol" | "mantis", boolean>> = {}
): CanProfileInventory {
  return {
    axol: {
      present: axol,
      up: axol,
      channels: { left: "can0", right: "can1" },
      automaticConnectSuppressed: suppressed.axol,
    },
    mantis: {
      present: mantis,
      up: mantis,
      channels: { left: "can2", right: "can3" },
      automaticConnectSuppressed: suppressed.mantis,
    },
  }
}

describe("chooseDetectedHardwareProfile", () => {
  it("moves a Mantis selection to Axol when only Axol is attached", () => {
    expect(chooseDetectedHardwareProfile(inventory(true, false), "mantis")).toBe("axol")
  })

  it("moves an Axol selection to Mantis when only Mantis is attached", () => {
    expect(chooseDetectedHardwareProfile(inventory(false, true), "axol")).toBe("mantis")
  })

  it("leaves a selection that already matches the attached device", () => {
    expect(chooseDetectedHardwareProfile(inventory(true, false), "axol")).toBeNull()
    expect(chooseDetectedHardwareProfile(inventory(false, true), "mantis")).toBeNull()
  })

  it("leaves the selection alone when the inventory is ambiguous", () => {
    expect(chooseDetectedHardwareProfile(inventory(true, true), "axol")).toBeNull()
    expect(chooseDetectedHardwareProfile(inventory(true, true), "mantis")).toBeNull()
    expect(chooseDetectedHardwareProfile(inventory(false, false), "axol")).toBeNull()
    expect(chooseDetectedHardwareProfile(inventory(false, false), "mantis")).toBeNull()
  })

  it("follows the attached device even when its auto-connect is suppressed", () => {
    expect(chooseDetectedHardwareProfile(inventory(true, false, { axol: true }), "mantis")).toBe(
      "axol"
    )
  })
})

describe("hardwarePresenceSignature", () => {
  it("changes only when Axol / Mantis presence changes", () => {
    const signatures = [
      inventory(false, false),
      inventory(true, false),
      inventory(false, true),
      inventory(true, true),
    ].map(hardwarePresenceSignature)
    expect(new Set(signatures).size).toBe(4)
    expect(hardwarePresenceSignature(inventory(true, false, { axol: true }))).toBe(signatures[1])
  })
})

describe("robotLinkReleasable", () => {
  it("frees a disconnected or failed link", () => {
    expect(robotLinkReleasable("disconnected", "axol", inventory(true, false))).toBe(true)
    expect(robotLinkReleasable("error", "mantis", null)).toBe(true)
  })

  it("frees a link still connected to a device that was unplugged", () => {
    // Mantis swapped for Axol, and the reverse.
    expect(robotLinkReleasable("connected", "mantis", inventory(true, false))).toBe(true)
    expect(robotLinkReleasable("connected", "axol", inventory(false, true))).toBe(true)
    expect(robotLinkReleasable("connected", "mantis", inventory(false, false))).toBe(true)
  })

  it("keeps a link connected to a device that is still plugged in", () => {
    expect(robotLinkReleasable("connected", "axol", inventory(true, false))).toBe(false)
    expect(robotLinkReleasable("connected", "mantis", inventory(true, true))).toBe(false)
    // Older hosts omit the profile, which means Axol.
    expect(robotLinkReleasable("connected", undefined, inventory(true, false))).toBe(false)
  })

  it("never takes a connected link without inventory, a busy link or one connecting", () => {
    expect(robotLinkReleasable("connected", "mantis", null)).toBe(false)
    expect(robotLinkReleasable("busy", "mantis", inventory(true, false))).toBe(false)
    expect(robotLinkReleasable("connecting", "mantis", inventory(true, false))).toBe(false)
    expect(robotLinkReleasable(undefined, undefined, inventory(true, false))).toBe(false)
  })
})

describe("shouldStartCanDiscovery", () => {
  const needed = { status: "needed", candidateCount: 0, generation: 1 } as const

  it("starts only when discovery is needed and the link, polls and hardware allow it", () => {
    expect(shouldStartCanDiscovery(needed, true, true, true)).toBe(true)
    expect(shouldStartCanDiscovery(needed, false, true, true)).toBe(false)
    expect(shouldStartCanDiscovery(needed, true, false, true)).toBe(false)
    expect(shouldStartCanDiscovery(needed, true, true, false)).toBe(false)
    expect(shouldStartCanDiscovery({ ...needed, status: "ready" }, true, true, true)).toBe(false)
    expect(shouldStartCanDiscovery(null, true, true, true)).toBe(false)
  })
})
