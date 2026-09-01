import assert from "node:assert/strict"
import test from "node:test"

import {
  autoConnectPollStateKnown,
  autoConnectRetryDelay,
  autoConnectSignature,
  canDiscoveryAttemptSignature,
  canDiscoveryBlocksAutoConnect,
  chooseAutoConnectProfile,
  chooseAutoConnectTarget,
  chooseDiagnosticsAutoConnectProfile,
  chooseUnambiguousAutoConnectProfile,
  nextAutoConnectAttempt,
  shouldStartCanDiscovery,
} from "../src/lib/can-auto-connect.ts"
import {
  ApiRequestError,
  canDiscoveryRequestCanRetry,
  discoverCanHardware,
  robotConnect,
  setServerBase,
} from "../src/lib/supervisor.ts"

function presence(left, right, present, up = present, automaticConnectSuppressed = false) {
  return { channels: { left, right }, present, up, automaticConnectSuppressed }
}

function discovery(status, generation = 4) {
  return { status, candidateCount: 1, generation }
}

test("CAN discovery blocks auto-connect while pending, running, or uncertain after error", () => {
  assert.equal(canDiscoveryBlocksAutoConnect(undefined), false)
  assert.equal(canDiscoveryBlocksAutoConnect(discovery("needed")), true)
  assert.equal(canDiscoveryBlocksAutoConnect(discovery("running")), true)
  assert.equal(canDiscoveryBlocksAutoConnect(discovery("error")), true)
  for (const status of ["ready", "configured", "partial", "unidentified"]) {
    assert.equal(canDiscoveryBlocksAutoConnect(discovery(status)), false)
  }
})

test("CAN discovery starts only from a known, idle, releasable link snapshot", () => {
  const needed = discovery("needed")
  assert.equal(shouldStartCanDiscovery(needed, "disconnected", true, true), true)
  assert.equal(shouldStartCanDiscovery(needed, "error", true, true), true)
  assert.equal(shouldStartCanDiscovery(needed, "connected", true, true), false)
  assert.equal(shouldStartCanDiscovery(needed, "disconnected", false, true), false)
  assert.equal(shouldStartCanDiscovery(needed, "disconnected", true, false), false)
  assert.equal(
    shouldStartCanDiscovery(discovery("unidentified"), "disconnected", true, true),
    false
  )
})

test("CAN discovery latches by the opaque server hardware generation", () => {
  assert.equal(canDiscoveryAttemptSignature(discovery("needed", 17)), "17")
  assert.equal(canDiscoveryAttemptSignature(discovery("running", 17)), null)
  assert.equal(canDiscoveryAttemptSignature(undefined), null)
})

test("chooses the only configured hardware profile that is present", () => {
  const profiles = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", false),
    mantis: presence("can_mantis_l", "can_mantis_r", true),
  }

  assert.equal(chooseAutoConnectProfile(profiles, "axol"), "mantis")
  assert.equal(chooseAutoConnectProfile(profiles, "mantis"), "mantis")
})

test("uses the selected operation as the tie-breaker when both are present", () => {
  const profiles = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", true),
    mantis: presence("can_mantis_l", "can_mantis_r", true),
  }

  assert.equal(chooseAutoConnectProfile(profiles, "axol"), "axol")
  assert.equal(chooseAutoConnectProfile(profiles, "mantis"), "mantis")
  assert.equal(chooseUnambiguousAutoConnectProfile(profiles), null)
})

test("a server-wide manual disconnect pauses only the same selected target", () => {
  const profiles = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", true, true, true),
    mantis: presence("can_mantis_l", "can_mantis_r", true),
  }

  assert.equal(chooseAutoConnectProfile(profiles, "axol"), null)
  assert.equal(chooseAutoConnectProfile(profiles, "mantis"), "mantis")
})

test("Diagnostics respects a manual disconnect of its one detected profile", () => {
  const profiles = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", true, true, true),
    mantis: presence("can_mantis_l", "can_mantis_r", false),
  }

  assert.equal(chooseUnambiguousAutoConnectProfile(profiles), null)
})

test("Diagnostics only auto-selects an unambiguous single present profile", () => {
  const onlyAxol = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", true),
    mantis: presence("can_mantis_l", "can_mantis_r", false),
  }
  const onlyMantis = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", false),
    mantis: presence("can_mantis_l", "can_mantis_r", true),
  }
  const neither = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", false),
    mantis: presence("can_mantis_l", "can_mantis_r", false),
  }

  assert.equal(chooseUnambiguousAutoConnectProfile(onlyAxol), "axol")
  assert.equal(chooseUnambiguousAutoConnectProfile(onlyMantis), "mantis")
  assert.equal(chooseUnambiguousAutoConnectProfile(neither), null)
})

test("Diagnostics preserves only the server-reported profile on legacy hosts", () => {
  assert.equal(chooseDiagnosticsAutoConnectProfile(null, true, "axol"), "axol")
  assert.equal(chooseDiagnosticsAutoConnectProfile(null, true, "mantis"), "mantis")
  assert.equal(chooseDiagnosticsAutoConnectProfile(null, false, "axol"), null)
})

test("does not guess when neither configured profile is present", () => {
  const profiles = {
    axol: presence("can_alm_axol_l", "can_alm_axol_r", false),
    mantis: presence("can_mantis_l", "can_mantis_r", false),
  }

  assert.equal(chooseAutoConnectProfile(profiles, "axol"), null)
})

test("moves a connected link off an obsolete map even when no replacement is present", () => {
  const profiles = {
    axol: presence("new-left", "new-right", false),
    mantis: presence("rig-left", "rig-right", false),
  }

  assert.equal(
    chooseAutoConnectTarget(
      profiles,
      "axol",
      "axol",
      { left: "old-left", right: "old-right" },
      true
    ),
    "axol"
  )
  assert.equal(
    chooseAutoConnectTarget(
      profiles,
      "axol",
      "axol",
      { left: "new-left", right: "new-right" },
      true
    ),
    null
  )
})

test("ordinary absence never starts a disconnected stale-map correction", () => {
  const profiles = {
    axol: presence("new-left", "new-right", false),
    mantis: presence("rig-left", "rig-right", false),
  }

  assert.equal(
    chooseAutoConnectTarget(
      profiles,
      "axol",
      "axol",
      { left: "old-left", right: "old-right" },
      false
    ),
    null
  )
})

test("present but down interfaces remain eligible for connection", () => {
  const profiles = {
    axol: presence("bench-left", null, true, false),
    mantis: presence("rig-left", "rig-right", false),
  }

  assert.equal(chooseAutoConnectProfile(profiles, "mantis"), "axol")
})

test("mapping changes produce a fresh auto-connect latch signature", () => {
  const original = presence("rig-left", "rig-right", true)
  const swapped = presence("rig-right", "rig-left", true)

  assert.notEqual(autoConnectSignature("mantis", original), autoConnectSignature("mantis", swapped))
})

test("an interface becoming administratively up permits a fresh attempt", () => {
  const down = presence("rig-left", "rig-right", true, false)
  const up = presence("rig-left", "rig-right", true, true)

  assert.notEqual(autoConnectSignature("mantis", down), autoConnectSignature("mantis", up))
})

test("automatic connection retries are bounded and backed off", () => {
  assert.equal(autoConnectRetryDelay(1), 2000)
  assert.equal(autoConnectRetryDelay(2), 5000)
  assert.equal(autoConnectRetryDelay(3), null)
  assert.equal(autoConnectRetryDelay(4), null)
})

test("an exhausted signature cannot be invoked again after inventory oscillates", () => {
  assert.equal(nextAutoConnectAttempt(0), 1)
  assert.equal(nextAutoConnectAttempt(1), 2)
  assert.equal(nextAutoConnectAttempt(2), 3)
  assert.equal(nextAutoConnectAttempt(3), null)
  assert.equal(nextAutoConnectAttempt(4), null)
})

test("unknown inventory or session state never consumes an automatic retry", () => {
  assert.equal(autoConnectPollStateKnown(true, true, true), true)
  assert.equal(autoConnectPollStateKnown(true, false, true), false)
  assert.equal(autoConnectPollStateKnown(true, true, false), false)

  const priorAttempts = 1
  assert.equal(nextAutoConnectAttempt(priorAttempts, false), null)
  assert.equal(nextAutoConnectAttempt(priorAttempts, true), 2)
})

test("CAN discovery uses the dedicated non-interactive backend action", async (context) => {
  setServerBase("")
  const inventory = {
    interfaces: [],
    profiles: {},
    discovery: discovery("configured", 8),
  }
  const fetchMock = context.mock.method(globalThis, "fetch", async () => Response.json(inventory))

  assert.deepEqual(await discoverCanHardware(), inventory)
  assert.equal(fetchMock.mock.callCount(), 1)
  assert.equal(fetchMock.mock.calls[0].arguments[0], "/api/can/discover")
  assert.deepEqual(fetchMock.mock.calls[0].arguments[1], { method: "POST" })
})

test("CAN discovery retries only transient request failures", () => {
  assert.equal(canDiscoveryRequestCanRetry(new ApiRequestError("busy", 409)), true)
  assert.equal(canDiscoveryRequestCanRetry(new ApiRequestError("failed", 500)), true)
  assert.equal(canDiscoveryRequestCanRetry(new TypeError("network failed")), true)
  assert.equal(canDiscoveryRequestCanRetry(new ApiRequestError("root required", 403)), false)
  assert.equal(canDiscoveryRequestCanRetry(new Error("invalid response")), false)
})

test("robot connect identifies automatic and manual requests to the backend", async (context) => {
  setServerBase("")
  const requests = []
  const fetchMock = context.mock.method(globalThis, "fetch", async (url, init) => {
    requests.push({ url, body: JSON.parse(init.body) })
    return new Response(
      JSON.stringify({
        state: "connected",
        connected: true,
        error: null,
        lastPing: null,
        motors: [],
        motorCount: 0,
        reachableCount: 0,
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    )
  })

  await robotConnect(undefined, "mantis", true)
  await robotConnect(undefined, "axol")

  assert.equal(fetchMock.mock.callCount(), 2)
  assert.deepEqual(requests, [
    {
      url: "/api/robot/connect",
      body: {
        leftChannel: null,
        rightChannel: null,
        channelsSet: false,
        profile: "mantis",
        automatic: true,
      },
    },
    {
      url: "/api/robot/connect",
      body: {
        leftChannel: null,
        rightChannel: null,
        channelsSet: false,
        profile: "axol",
        automatic: false,
      },
    },
  ])
})
