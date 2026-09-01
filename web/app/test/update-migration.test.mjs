import assert from "node:assert/strict"
import test from "node:test"

import {
  FIRST_SAFE_SELF_UPDATE_VERSION,
  requiresInstallerMigration,
} from "../src/lib/update-migration.ts"

test("legacy releases require the one-time hosted-installer migration", () => {
  assert.equal(FIRST_SAFE_SELF_UPDATE_VERSION, "0.1.36")
  for (const version of [null, "", "unknown", "0.1.34", "0.1.35", "v0.1.35"]) {
    assert.equal(requiresInstallerMigration(version), true, String(version))
  }
})

test("the hardened release and later releases may self-update", () => {
  for (const version of ["0.1.36", "v0.1.36", "0.1.36.0", "0.1.37", "0.2.0", "1.0.0"]) {
    assert.equal(requiresInstallerMigration(version), false, version)
  }
})
