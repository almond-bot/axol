import { describe, expect, it } from "vitest"

import {
  deviceErrorText,
  discoveryNeedsRecheck,
  discoveryNotice,
  nextRecheckDelay,
} from "./hardware-status"

const discovery = (status: string) =>
  ({ status, candidateCount: 1, generation: 1, message: "CAN assignment not revalidated" }) as never

describe("discoveryNotice", () => {
  it("asks for power, not for a CAN retry", () => {
    const notice = discoveryNotice(discovery("unidentified"))
    expect(notice?.text).toContain("Turn on its power")
    expect(notice?.text).not.toMatch(/CAN|identif|retry/i)
  })

  it("says nothing once the hardware is known or while it is being checked", () => {
    for (const status of ["ready", "configured", "needed", "running"]) {
      expect(discoveryNotice(discovery(status))).toBeNull()
    }
    expect(discoveryNotice(null)).toBeNull()
  })

  it("keeps checking only hardware that hasn't answered", () => {
    expect(discoveryNeedsRecheck(discovery("unidentified"))).toBe(true)
    expect(discoveryNeedsRecheck(discovery("error"))).toBe(true)
    expect(discoveryNeedsRecheck(discovery("configured"))).toBe(false)
  })
})

describe("nextRecheckDelay", () => {
  it("starts at 5 s and backs off to a minute", () => {
    const delays = [null, 5000, 10_000, 20_000, 40_000, 60_000].map(nextRecheckDelay)
    expect(delays).toEqual([5000, 10_000, 20_000, 40_000, 60_000, 60_000])
  })
})

describe("deviceErrorText", () => {
  it.each([
    ["Robot not detected", "Axol isn't plugged in. Check its USB cable."],
    ["Error: No CAN interfaces configured", "Axol isn't plugged in. Check its USB cable."],
    ["Left arm timed out on can_alm_axol_l", "Axol isn't answering. Turn on its power."],
    [
      "The saved Axol CAN assignment could not be revalidated. Power its Axol motors",
      "Axol isn't answering. Turn on its power.",
    ],
    [
      "PermissionError: [Errno 1] Operation not permitted",
      "The robot's computer can't open the USB device. Restart the robot's computer.",
    ],
    ["OSError: something odd", "Couldn't connect to Axol."],
    [null, "Couldn't connect to Axol."],
  ])("%s", (raw, text) => {
    expect(deviceErrorText(raw, "Axol")).toBe(text)
  })
})
