import assert from "node:assert/strict"
import test from "node:test"

import { probeUpdateStatus, setServerBase } from "../src/lib/supervisor.ts"

test("a missing update endpoint identifies the unsafe main-tracking servers", async (context) => {
  setServerBase("")
  const fetchMock = context.mock.method(globalThis, "fetch", async (url) => {
    assert.equal(url, "/api/update/status")
    return new Response('{"detail":"Not Found"}', {
      status: 404,
      headers: { "content-type": "application/json" },
    })
  })

  assert.equal(await probeUpdateStatus(), null)
  assert.equal(fetchMock.mock.callCount(), 1)
})

test("a modern update endpoint returns its status without touching another route", async (context) => {
  setServerBase("")
  const status = {
    enabled: true,
    version: "0.1.36",
    remoteVersion: "0.1.36",
    updateAvailable: false,
    idle: true,
    state: "idle",
    phase: null,
    error: null,
  }
  const fetchMock = context.mock.method(globalThis, "fetch", async (url) => {
    assert.equal(url, "/api/update/status")
    return new Response(JSON.stringify(status), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  })

  assert.deepEqual(await probeUpdateStatus(), status)
  assert.equal(fetchMock.mock.callCount(), 1)
})
