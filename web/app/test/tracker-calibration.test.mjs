import assert from "node:assert/strict"
import test from "node:test"

import { hasApprovedTrackerFactoryTransform } from "../src/lib/tracker-calibration.ts"

test("only approved tracker conventions promise a factory fallback", () => {
  assert.equal(hasApprovedTrackerFactoryTransform("lighthouse", null), true)
  assert.equal(
    hasApprovedTrackerFactoryTransform("ultimate", { quatOrder: "wxyz", upAxis: "z" }),
    true
  )
  assert.equal(
    hasApprovedTrackerFactoryTransform("ultimate", { quatOrder: "xyzw", upAxis: "z" }),
    false
  )
  assert.equal(
    hasApprovedTrackerFactoryTransform("ultimate", { quatOrder: "wxyz", upAxis: "y" }),
    false
  )
  assert.equal(hasApprovedTrackerFactoryTransform("ultimate", null), false)
  assert.equal(hasApprovedTrackerFactoryTransform("quest", null), false)
})
