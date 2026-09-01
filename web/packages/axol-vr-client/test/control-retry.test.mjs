import assert from "node:assert/strict"
import test from "node:test"

import {
  controlPeerNeedsRetry,
  controlRequestIsDue,
  retireCurrentControlPeer,
} from "../src/controlRetry.ts"

test("a missing control offer is retried until one is seen", () => {
  assert.equal(controlRequestIsDue(false, 1_000, 3_999, 3_000), false)
  assert.equal(controlRequestIsDue(false, 1_000, 4_000, 3_000), true)
  assert.equal(controlRequestIsDue(true, 1_000, 10_000, 3_000), false)
})

test("closed data channels and terminal peer states require a retry", () => {
  for (const state of ["disconnected", "failed", "closed"]) {
    assert.equal(controlPeerNeedsRetry(state), true, state)
  }
  for (const state of ["new", "connecting", "connected"]) {
    assert.equal(controlPeerNeedsRetry(state), false, state)
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
