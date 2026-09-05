import assert from "node:assert/strict"
import test from "node:test"

import {
  AXOL_CATEGORY_KEYS,
  SETTINGS_SCOPES,
  defaultSettingsTab,
  settingsScopeForTab,
} from "../src/lib/settings-scope.ts"

test("every connection tile has a settings scope with a landing tab in that scope", () => {
  assert.deepEqual(
    SETTINGS_SCOPES.map((s) => s.key),
    ["axol", "mantis", "general"]
  )
  for (const { key } of SETTINGS_SCOPES) {
    assert.equal(settingsScopeForTab(defaultSettingsTab(key)), key, key)
  }
})

test("Axol tabs are the arm hardware and behaviour categories", () => {
  for (const tab of ["cameras", "pose", ...AXOL_CATEGORY_KEYS]) {
    assert.equal(settingsScopeForTab(tab), "axol", tab)
  }
})

test("Mantis tabs cover tracking, CAN mapping, and wrist cameras", () => {
  for (const tab of ["mantis-tracking", "mantis-can", "mantis-cameras"]) {
    assert.equal(settingsScopeForTab(tab), "mantis", tab)
  }
})

test("shared and unknown categories land under General", () => {
  for (const tab of ["usb", "recording", "inference", "system", "advanced", "a-new-category"]) {
    assert.equal(settingsScopeForTab(tab), "general", tab)
  }
})
