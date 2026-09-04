import assert from "node:assert/strict"
import test from "node:test"

import {
  FIRST_SAFE_SELF_UPDATE_VERSION,
  requiresInstallerMigration,
  showInstallerMigration,
} from "../src/lib/update-migration.ts"

test("legacy releases require the one-time hosted-installer migration", () => {
  assert.equal(FIRST_SAFE_SELF_UPDATE_VERSION, "0.1.37")
  for (const version of [null, "", "unknown", "0.1.34", "0.1.35", "v0.1.35", "0.1.36", "v0.1.36"]) {
    assert.equal(requiresInstallerMigration(version), true, String(version))
  }
})

test("the hardened release and later releases may self-update", () => {
  for (const version of ["0.1.37", "v0.1.37", "0.1.37.0", "0.1.38", "0.2.0", "1.0.0"]) {
    assert.equal(requiresInstallerMigration(version), false, version)
  }
})

test("an unidentified release install still shows the fail-closed migration", () => {
  assert.equal(showInstallerMigration(null, true), true)
  assert.equal(showInstallerMigration(null, false), false)
  assert.equal(showInstallerMigration("0.1.36", true), true)
  assert.equal(showInstallerMigration("0.1.37", true), false)
})
