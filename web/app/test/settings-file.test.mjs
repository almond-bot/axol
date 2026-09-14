import assert from "node:assert/strict"
import test from "node:test"

import {
  SETTINGS_FILE_VERSION,
  buildSettingsFile,
  flattenSettings,
  nestSettings,
  parseSettingsFile,
} from "../src/lib/settings-file.ts"

const flat = {
  "axol.left_stiffness": 0.8,
  "axol.left.elbow.kp": 60,
  "teleop.rest_pose_left": [0, 0.1, 0, 0, 0, 0, 0],
  "robot.right_channel": "null",
  "axol.has_gripper": false,
}

test("nesting groups keys by section and keeps arrays as leaves", () => {
  assert.deepEqual(nestSettings(flat), {
    axol: { has_gripper: false, left: { elbow: { kp: 60 } }, left_stiffness: 0.8 },
    robot: { right_channel: "null" },
    teleop: { rest_pose_left: [0, 0.1, 0, 0, 0, 0, 0] },
  })
  assert.deepEqual(flattenSettings(nestSettings(flat)), flat)
})

test("an exported file round-trips and carries the version and cameras", () => {
  const cameras = { serials: { overhead: "123" } }
  const file = buildSettingsFile(flat, cameras)
  assert.equal(file.version, SETTINGS_FILE_VERSION)
  assert.deepEqual(file.cameras, cameras)
  assert.deepEqual(parseSettingsFile(JSON.parse(JSON.stringify(file))), { values: flat, cameras })
})

test("a pre-v2 export still imports, curated values over advanced", () => {
  const legacy = {
    values: { "robot.left_stiffness": 0.7, "teleop.frequency": 240 },
    advanced: { "axol.left.elbow.kp": 60, "vr_teleop.frequency": 120 },
    cameras: null,
  }
  assert.deepEqual(parseSettingsFile(legacy), {
    values: {
      "axol.left.elbow.kp": 60,
      "vr_teleop.frequency": 120,
      "robot.left_stiffness": 0.7,
      "teleop.frequency": 240,
    },
    cameras: null,
  })
})

test("garbage is rejected and non-object sections are ignored", () => {
  assert.throws(() => parseSettingsFile("nope"), /invalid settings file/)
  assert.deepEqual(parseSettingsFile({ version: 2, axol: 5, teleop: { frequency: 1 } }), {
    values: { "teleop.frequency": 1 },
    cameras: undefined,
  })
})
