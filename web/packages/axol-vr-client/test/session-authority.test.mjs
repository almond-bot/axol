import assert from "node:assert/strict"
import test from "node:test"

import { webxrCanControlPose, webxrCanControlRecording } from "../src/sessionAuthority.ts"

test("tracker-owned sessions keep WebXR recording controls view-only", () => {
  assert.equal(webxrCanControlRecording("data_collection", "tracker"), false)
})

test("Quest-owned and legacy collection sessions retain recording controls", () => {
  assert.equal(webxrCanControlRecording("data_collection", "webxr"), true)
  assert.equal(webxrCanControlRecording(null, null), true)
})

test("teleop never exposes recording controls", () => {
  assert.equal(webxrCanControlRecording("teleop", "webxr"), false)
  assert.equal(webxrCanControlRecording("teleop", null), false)
})

test("Quest teleop retains pose reset while tracker viewers cannot control it", () => {
  assert.equal(webxrCanControlPose("webxr"), true)
  assert.equal(webxrCanControlPose(null), true)
  assert.equal(webxrCanControlPose("tracker"), false)
})
