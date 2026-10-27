import { describe, expect, it } from "vitest"

import { camerasSettingsTab, defaultSettingsTab, settingsScopeForTab } from "./settings-scope"

describe("camerasSettingsTab", () => {
  it("opens each device's own camera layout, in that device's scope", () => {
    expect(camerasSettingsTab("axol")).toBe("cameras")
    expect(camerasSettingsTab("mantis")).toBe("mantis-cameras")
    expect(settingsScopeForTab(camerasSettingsTab("mantis"))).toBe("mantis")
    // Mantis opens on Tracking from its tile, so the cameras prompt can't reuse it.
    expect(camerasSettingsTab("mantis")).not.toBe(defaultSettingsTab("mantis"))
  })
})
