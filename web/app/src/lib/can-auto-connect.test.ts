import { describe, expect, it } from "vitest"

import { chooseDetectedHardwareProfile, hardwarePresenceSignature } from "./can-auto-connect"
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
