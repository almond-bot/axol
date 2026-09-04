import assert from "node:assert/strict"
import test from "node:test"

import {
  retireCurrentVideoPeer,
  videoDisconnectRearmDelay,
  videoPeerNeedsRetry,
  videoRequestIsDue,
} from "../src/videoRetry.ts"

test("only terminal peer states require an immediate retry", () => {
  for (const state of ["failed", "closed"]) {
    assert.equal(videoPeerNeedsRetry(state), true, state)
  }
  // "disconnected" is transient (TURN hiccup / Wi-Fi roam) and gets a grace
  // period instead of an immediate teardown + renegotiation.
  for (const state of ["new", "connecting", "connected", "disconnected"]) {
    assert.equal(videoPeerNeedsRetry(state), false, state)
  }
})

test("a disconnected peer is re-armed only once its grace period elapses", () => {
  const disconnectedAt = 10_000
  // Freshly disconnected: wait out the whole grace.
  assert.equal(videoDisconnectRearmDelay("disconnected", disconnectedAt, 10_000, 5_000), 5_000)
  // Partway through: wait the remainder (a timer that fired early reschedules).
  assert.equal(videoDisconnectRearmDelay("disconnected", disconnectedAt, 13_000, 5_000), 2_000)
  assert.equal(videoDisconnectRearmDelay("disconnected", disconnectedAt, 14_999, 5_000), 1)
  // Grace elapsed and still disconnected: due now.
  assert.equal(videoDisconnectRearmDelay("disconnected", disconnectedAt, 15_000, 5_000), 0)
  assert.equal(videoDisconnectRearmDelay("disconnected", disconnectedAt, 20_000, 5_000), 0)
})

test("a peer that recovered before the grace elapsed is not re-armed", () => {
  for (const state of ["connected", "completed", "connecting", "new"]) {
    assert.equal(videoDisconnectRearmDelay(state, 10_000, 15_000, 5_000), null, state)
  }
  // Terminal states are the immediate-retry path, not the grace path.
  for (const state of ["failed", "closed"]) {
    assert.equal(videoDisconnectRearmDelay(state, 10_000, 15_000, 5_000), null, state)
  }
})

test("a missing video offer is retried at the request cadence", () => {
  // lastRequestAt = 0 means "never sent": the first request goes out at once.
  assert.equal(videoRequestIsDue(false, 0, 1_000_000, 2_000), true)
  assert.equal(videoRequestIsDue(false, 1_000, 2_999, 2_000), false)
  assert.equal(videoRequestIsDue(false, 1_000, 3_000, 2_000), true)
  assert.equal(videoRequestIsDue(true, 1_000, 10_000, 2_000), false)
})

test("a failed negotiation backs off instead of re-requesting on the next tick", () => {
  // rearmPeer zeroes lastRequestAt (cadence satisfied) but sets notBefore =
  // failure time + backoff; the 300 ms poll must not fire until then.
  const failedAt = 5_000
  const notBefore = failedAt + 2_000
  assert.equal(videoRequestIsDue(false, 0, failedAt + 300, 2_000, notBefore), false)
  assert.equal(videoRequestIsDue(false, 0, notBefore - 1, 2_000, notBefore), false)
  assert.equal(videoRequestIsDue(false, 0, notBefore, 2_000, notBefore), true)
  // A socket swap clears the backoff so a fresh socket requests immediately.
  assert.equal(videoRequestIsDue(false, 0, failedAt + 300, 2_000, 0), true)
})

test("one peer terminal transition claims exactly one retry", () => {
  const socket = {}
  const peer = {}
  const socketRef = { current: socket }
  const peerRef = { current: peer }

  assert.equal(retireCurrentVideoPeer(socketRef, peerRef, socket, peer), true)
  assert.equal(peerRef.current, null)
  assert.equal(retireCurrentVideoPeer(socketRef, peerRef, socket, peer), false)
})

test("late terminal callback cannot retire a replacement peer or socket", () => {
  const oldSocket = {}
  const newSocket = {}
  const oldPeer = {}
  const newPeer = {}
  const socketRef = { current: newSocket }
  const peerRef = { current: newPeer }

  assert.equal(retireCurrentVideoPeer(socketRef, peerRef, oldSocket, oldPeer), false)
  assert.equal(retireCurrentVideoPeer(socketRef, peerRef, newSocket, oldPeer), false)
  assert.equal(peerRef.current, newPeer)
})

test("terminal callback after teardown cannot re-arm", () => {
  const socket = {}
  const peer = {}
  const socketRef = { current: null }
  const peerRef = { current: null }

  assert.equal(retireCurrentVideoPeer(socketRef, peerRef, socket, peer), false)
})
