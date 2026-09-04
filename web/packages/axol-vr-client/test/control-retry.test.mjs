import assert from "node:assert/strict"
import test from "node:test"

import {
  controlDisconnectRearmDelay,
  controlPeerNeedsRetry,
  controlRequestIsDue,
  retireCurrentControlPeer,
} from "../src/controlRetry.ts"

test("a missing control offer is retried until one is seen", () => {
  assert.equal(controlRequestIsDue(false, 1_000, 3_999, 3_000), false)
  assert.equal(controlRequestIsDue(false, 1_000, 4_000, 3_000), true)
  assert.equal(controlRequestIsDue(true, 1_000, 10_000, 3_000), false)
})

test("a failed negotiation backs off instead of re-requesting on the next tick", () => {
  // rearmPeer/rearmSocket zero lastRequestAt but set notBefore = failure time +
  // backoff, so a fast deterministic failure can't loop every 300 ms.
  const failedAt = 5_000
  const notBefore = failedAt + 3_000
  assert.equal(controlRequestIsDue(false, 0, failedAt + 300, 3_000, notBefore), false)
  assert.equal(controlRequestIsDue(false, 0, notBefore - 1, 3_000, notBefore), false)
  assert.equal(controlRequestIsDue(false, 0, notBefore, 3_000, notBefore), true)
  // A socket swap clears the backoff so a fresh socket requests immediately.
  assert.equal(controlRequestIsDue(false, 0, failedAt + 300, 3_000, 0), true)
  // An offer landing cancels the request regardless of the backoff.
  assert.equal(controlRequestIsDue(true, 0, notBefore + 1, 3_000, notBefore), false)
})

test("only terminal peer states require an immediate retry", () => {
  for (const state of ["failed", "closed"]) {
    assert.equal(controlPeerNeedsRetry(state), true, state)
  }
  // "disconnected" is transient and gets a grace period instead.
  for (const state of ["new", "connecting", "connected", "disconnected"]) {
    assert.equal(controlPeerNeedsRetry(state), false, state)
  }
})

test("a disconnected control peer is re-armed only once its grace elapses", () => {
  const disconnectedAt = 10_000
  assert.equal(controlDisconnectRearmDelay("disconnected", disconnectedAt, 10_000, 3_000), 3_000)
  assert.equal(controlDisconnectRearmDelay("disconnected", disconnectedAt, 12_000, 3_000), 1_000)
  assert.equal(controlDisconnectRearmDelay("disconnected", disconnectedAt, 13_000, 3_000), 0)
  assert.equal(controlDisconnectRearmDelay("disconnected", disconnectedAt, 30_000, 3_000), 0)
  for (const state of ["connected", "completed", "connecting", "new", "failed", "closed"]) {
    assert.equal(controlDisconnectRearmDelay(state, disconnectedAt, 13_000, 3_000), null, state)
  }
})

test("only the current control peer can claim a retry", () => {
  const socket = {}
  const peer = {}
  const socketRef = { current: socket }
  const peerRef = { current: peer }

  assert.equal(retireCurrentControlPeer(socketRef, peerRef, socket, peer), true)
  assert.equal(peerRef.current, null)
  assert.equal(retireCurrentControlPeer(socketRef, peerRef, socket, peer), false)

  const replacement = {}
  peerRef.current = replacement
  assert.equal(retireCurrentControlPeer(socketRef, peerRef, socket, peer), false)
  assert.equal(peerRef.current, replacement)
})
