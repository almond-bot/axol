import { describe, expect, it } from "vitest"

import { batteryLevel, batteryText, batteryTooltip, formatAge, jellyBattery } from "./jelly-view"
import type { JellyBattery, JellyStatus } from "./supervisor"

const battery = (over: Partial<JellyBattery> = {}): JellyBattery => ({
  voltage: 26.1,
  percent: 51.2,
  charging: false,
  underLoad: false,
  ageSeconds: 0.4,
  live: true,
  ...over,
})

const jelly = (state: string, b: JellyBattery | null | undefined): JellyStatus =>
  ({
    wheels: { state: "disconnected", motors: [], motorCount: 0, reachableCount: 0 },
    lift: { state, reachable: true, status: null, battery: b },
  }) as unknown as JellyStatus

describe("jellyBattery", () => {
  it("shows the reading while the lift link is up or lent to a task", () => {
    const b = battery()
    expect(jellyBattery(jelly("connected", b))).toBe(b)
    expect(jellyBattery(jelly("busy", b))).toBe(b)
  })

  it("hides it for a disconnected link, an older host, or no reading", () => {
    expect(jellyBattery(jelly("disconnected", battery()))).toBeNull()
    expect(jellyBattery(jelly("error", battery()))).toBeNull()
    expect(jellyBattery(jelly("connected", undefined))).toBeNull()
    expect(jellyBattery(jelly("connected", null))).toBeNull()
    expect(jellyBattery(null)).toBeNull()
  })
})

describe("battery display", () => {
  it("grades the level", () => {
    expect(batteryLevel(battery({ percent: 80 }))).toBe("ok")
    expect(batteryLevel(battery({ percent: 25 }))).toBe("low")
    expect(batteryLevel(battery({ percent: 10 }))).toBe("critical")
    expect(batteryLevel(battery({ percent: 100, charging: true }))).toBe("charging")
  })

  it("marks a reading taken under load as approximate", () => {
    expect(batteryText(battery())).toBe("51%")
    expect(batteryText(battery({ underLoad: true }))).toBe("~51%")
    expect(batteryText(battery({ charging: true }))).toBe("Charging")
  })

  it("explains the voltage, the reading's age, and load in the tooltip", () => {
    const live = batteryTooltip(battery())
    expect(live).toContain("Battery 51% · 26.10 V")
    expect(live).not.toContain("Last reading")
    const stale = batteryTooltip(battery({ live: false, ageSeconds: 750, underLoad: true }))
    expect(stale).toContain("Last reading 13 min ago")
    expect(stale).toContain("reads low")
  })

  it("formats ages", () => {
    expect(formatAge(12.4)).toBe("12 s")
    expect(formatAge(240)).toBe("4 min")
    expect(formatAge(3600)).toBe("1 h")
    expect(formatAge(7500)).toBe("2 h 5 min")
  })
})
