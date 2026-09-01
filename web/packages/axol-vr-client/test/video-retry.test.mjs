import assert from "node:assert/strict"
import test from "node:test"

import { retireCurrentVideoPeer, videoPeerNeedsRetry } from "../src/videoRetry.ts"

test("disconnected and terminal peer states require a retry", () => {
  for (const state of ["disconnected", "failed", "closed"]) {
    assert.equal(videoPeerNeedsRetry(state), true, state)
  }
  for (const state of ["new", "connecting", "connected"]) {
    assert.equal(videoPeerNeedsRetry(state), false, state)
  }
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
