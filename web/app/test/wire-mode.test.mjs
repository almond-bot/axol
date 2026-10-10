import assert from "node:assert/strict"
import test from "node:test"

import {
  MYACTUATOR_JOINTS,
  effectiveWireMode,
  parseA4Tokens,
  serializeA4Tokens,
  toggleA4Token,
} from "../src/lib/wire-mode.ts"

test("only MyActuator joints are offered the firmware loop", () => {
  assert.deepEqual(MYACTUATOR_JOINTS, ["shoulder_1", "shoulder_2", "shoulder_3", "elbow", "wrist_1"])
})

test("parse drops unknown tokens and serializes in a stable order", () => {
  const tokens = parseA4Tokens("right.shoulder_1  left.wrist_2 bogus left.shoulder_1 right.elbow")
  assert.deepEqual([...tokens].sort(), ["left.shoulder_1", "right.elbow", "right.shoulder_1"])
  assert.equal(serializeA4Tokens(tokens), "left.shoulder_1 right.shoulder_1 right.elbow")
  assert.equal(serializeA4Tokens(parseA4Tokens(undefined)), "")
})

test("toggle adds then removes a cell", () => {
  const on = toggleA4Token("", "right", "shoulder_1")
  assert.equal(on, "right.shoulder_1")
  assert.equal(toggleA4Token(on, "left", "shoulder_1"), "left.shoulder_1 right.shoulder_1")
  assert.equal(toggleA4Token(on, "right", "shoulder_1"), "")
})

test("the config's wire_mode a4 counts as firmware even when the form is empty", () => {
  const cfg = { right: { shoulder_1: "a4", shoulder_2: "mit" }, left: {} }
  assert.equal(effectiveWireMode("", cfg, "right", "shoulder_1"), "a4")
  assert.equal(effectiveWireMode("", cfg, "right", "shoulder_2"), "mit")
  assert.equal(effectiveWireMode("", null, "left", "elbow"), "mit")
  assert.equal(effectiveWireMode("left.elbow", null, "left", "elbow"), "a4")
})
