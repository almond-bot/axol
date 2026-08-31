import assert from "node:assert/strict"
import test from "node:test"

import { initialPoseSequence, nextPoseSequence } from "../src/poseSequence.ts"

function memoryStorage(initialReservation) {
  const values = new Map()
  if (initialReservation !== undefined) {
    values.set("axol.webxr.pose-sequence-reserved.v2", String(initialReservation))
  }
  return {
    getItem(key) {
      return values.get(key) ?? null
    },
    setItem(key, value) {
      values.set(key, value)
    },
  }
}

test("duplicated tabs reserve separate randomized sequence blocks", () => {
  // Duplicating a tab clones the same sessionStorage snapshot. Different page
  // randomness must still put the two producers in non-overlapping ranges.
  const first = initialPoseSequence({
    storage: memoryStorage(10_000_000),
    nowMs: () => 1,
    randomUint32: () => 17,
  })
  const duplicate = initialPoseSequence({
    storage: memoryStorage(10_000_000),
    nowMs: () => 1,
    randomUint32: () => 18,
  })

  assert.equal(first.reservedThrough, duplicate.current)
  assert.notEqual(nextPoseSequence(first), nextPoseSequence(duplicate))
})

test("reload starts beyond the range reserved by the previous page", () => {
  const storage = memoryStorage()
  const first = initialPoseSequence({
    storage,
    nowMs: () => 1,
    randomUint32: () => 0,
  })
  const reloaded = initialPoseSequence({
    storage,
    nowMs: () => 1,
    randomUint32: () => 0,
  })

  assert.ok(reloaded.current > first.reservedThrough)
  assert.ok(Number.isSafeInteger(reloaded.reservedThrough))
})
