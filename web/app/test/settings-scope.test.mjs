import assert from "node:assert/strict"
import test from "node:test"

import {
  AXOL_CATEGORY_KEYS,
  JELLY_CATEGORY_KEYS,
  JELLY_PARAMETERS_TAB,
  SETTINGS_SCOPES,
  defaultSettingsTab,
  settingsScopeForTab,
} from "../src/lib/settings-scope.ts"

test("every connection tile has a settings scope with a landing tab in that scope", () => {
  assert.deepEqual(
    SETTINGS_SCOPES.map((s) => s.key),
    ["axol", "mantis", "jelly", "general"]
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

test("Jelly tabs are the wheels / lift switches and the drive parameters", () => {
  for (const tab of [...JELLY_CATEGORY_KEYS, JELLY_PARAMETERS_TAB]) {
    assert.equal(settingsScopeForTab(tab), "jelly", tab)
  }
  // The Jelly switches are not Axol arm settings.
  for (const key of JELLY_CATEGORY_KEYS) assert.ok(!AXOL_CATEGORY_KEYS.has(key), key)
})

test("shared and unknown categories land under General", () => {
  for (const tab of ["usb", "recording", "inference", "system", "advanced", "a-new-category"]) {
    assert.equal(settingsScopeForTab(tab), "general", tab)
  }
})
