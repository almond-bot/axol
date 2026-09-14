import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import * as api from "./supervisor"
import * as telemetry from "./telemetry"

const session = {
  id: "s1",
  command: "teleop",
  args: {},
  status: "running",
  exitCode: null,
  error: null,
  startedAt: 1,
  pid: 2,
}

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("supervisor API client", () => {
  const requests: Array<{ url: string; init?: RequestInit }> = []

  beforeEach(() => {
    requests.length = 0
    api.setServerBase("robot.local")
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        requests.push({ url, init })
        if (url.endsWith("/api/info")) return response({ hostname: "axol" })
        if (url.includes("/api/update/status")) return response({ enabled: true })
        if (url.endsWith("/api/update/start")) return response({ started: true })
        if (url.endsWith("/api/host/shutdown") || url.endsWith("/api/host/restart")) {
          return response({ ok: true })
        }
        if (url.endsWith("/api/robot/status")) return response({ state: "connected" })
        if (url.endsWith("/api/robot/connect") || url.endsWith("/api/robot/disconnect")) {
          return response({ state: "connected" })
        }
        if (url.endsWith("/api/can/interfaces")) return response({ interfaces: [] })
        if (url.endsWith("/api/cameras/detect")) return response({ devices: [], error: null })
        if (url.endsWith("/api/cameras/restart-daemon")) return response({ ok: true, error: null })
        if (url.endsWith("/api/usb/status") || url.endsWith("/api/usb/connect")) {
          return response({ ready: true })
        }
        if (url.endsWith("/api/usb/proximity")) return response({ ok: true })
        if (url.endsWith("/api/op/status")) return response({ running: false, policy: null })
        if (url.endsWith("/api/op/start") || url.endsWith("/api/op/stop")) return response(session)
        if (url.endsWith("/api/op/episode")) return response({ ok: true })
        if (url.endsWith("/api/datasets")) return response({ datasets: [{ repoId: "org/data" }] })
        if (url.endsWith("/api/commands")) return response([])
        if (url.endsWith("/api/sessions")) return response([session])
        if (url.endsWith("/api/run") || url.includes("/api/sessions/s1/stop")) {
          return response(session)
        }
        if (url.includes("/api/sessions/s1/input")) return response({ ok: true })
        if (url.endsWith("/api/settings")) {
          return response({ values: {}, cameras: null, advanced: {} })
        }
        return response({})
      })
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    api.setServerBase("")
  })

  it("calls read-only status endpoints and normalizes their payloads", async () => {
    expect(await api.fetchInfo()).toEqual({ hostname: "axol" })
    expect(await api.fetchUpdateStatus(true)).toEqual({ enabled: true })
    expect(await api.fetchRobotStatus()).toEqual({ state: "connected" })
    expect(await api.fetchCanInterfaces()).toEqual({ interfaces: [] })
    expect(await api.detectCameras()).toEqual({ devices: [], error: null })
    expect(await api.fetchUsbStatus()).toEqual({ ready: true })
    expect(await api.fetchOpStatus()).toEqual({ running: false, policy: null })
    expect(await api.fetchDatasets()).toEqual([{ repoId: "org/data" }])
    expect(await api.fetchCommands()).toEqual([])
    expect(await api.fetchSessions()).toEqual([session])
    expect(await api.fetchSettings()).toEqual({ values: {}, cameras: null, advanced: {} })
    expect(requests.every((item) => item.url.startsWith("https://robot.local:8001"))).toBe(true)
  })

  it("sends the complete mutation contract", async () => {
    await api.startUpdate()
    await api.shutdownHost()
    await api.restartHost()
    await api.robotConnect({ left: "can0", right: null })
    await api.robotDisconnect()
    await api.restartCameraDaemon()
    await api.usbConnect()
    await api.setQuestProximityDisabled(true)
    await api.startOperation("teleop", { sim: true })
    await api.stopOperation()
    await api.sendEpisodeCommand("save")
    await api.runCommand("motor.info", { arm: "left" })
    await api.stopSession("s1")
    await api.sendSessionInput("s1", "continue")
    const saved = await api.saveSettings({ values: { "robot.has_gripper": false } })

    expect(saved.schema).toEqual([])
    expect(saved.advancedSchema).toEqual([])
    expect(
      requests.every((item) => item.init?.method === "POST" || item.init?.method === "PUT")
    ).toBe(true)
    expect(requests.find((item) => item.url.endsWith("/api/robot/connect"))?.init?.body).toContain(
      '"channelsSet":true'
    )
    expect(requests.find((item) => item.url.endsWith("/api/op/start"))?.init?.body).toContain(
      '"cameras":null'
    )
  })

  it("surfaces server and invalid-host responses as useful errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({ error: "robot busy" }, 409))
    )
    await expect(api.fetchInfo()).rejects.toThrow("robot busy")

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("not json"))
    )
    await expect(api.fetchInfo()).rejects.toThrow("did not answer like an axol serve host")
  })

  it("turns connection probe timeouts into a concise offline error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new DOMException("late", "TimeoutError")))
    )
    await expect(api.fetchCommands()).rejects.toThrow("no response from the host after 2s")
  })

  it("builds URLs and fault labels", () => {
    expect(api.apiUrl("/api/info")).toBe("https://robot.local:8001/api/info")
    expect(api.wsBaseUrl()).toBe("wss://robot.local:8001")
    expect(api.urdfUrl().endsWith("/api/urdf/axol.urdf")).toBe(true)
    expect(
      api.motorFaultLabel({
        arm: "left",
        joint: "SHOULDER_1",
        problem: "over temperature",
        temperature: 77.6,
      })
    ).toBe("L shoulder 1 — over temperature (78°C)")
  })
})

describe("telemetry API client", () => {
  beforeEach(() => {
    api.setServerBase("")
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.includes("/history")) return response({ frames: [] })
        if (url.includes("/motors/")) return response({ arm: "left", joint: "ELBOW" })
        if (url.endsWith("/api/diagnostics/run")) return response({ run: null, session })
        if (url.endsWith("/api/diagnostics/runs")) return response({ runs: [], removed: 2 })
        if (url.includes("/api/diagnostics/runs/")) {
          return response({ meta: { id: "run1" }, frames: [], log: [] })
        }
        return response({ state: "connected", sampleHz: 10, slow: {}, slowT: null, latest: null })
      })
    )
  })

  afterEach(() => vi.unstubAllGlobals())

  it("covers telemetry, motor, and diagnostic run endpoints", async () => {
    expect((await telemetry.fetchTelemetry()).sampleHz).toBe(10)
    expect(await telemetry.fetchTelemetryHistory(60, 20)).toEqual({ frames: [] })
    expect(await telemetry.fetchMotorDetails("left", "ELBOW")).toMatchObject({ joint: "ELBOW" })
    expect(await telemetry.startDiagnosticsRun("rom-test", { arm: "left" })).toMatchObject({
      run: null,
    })
    expect(await telemetry.fetchDiagnosticsRuns()).toEqual({ runs: [], removed: 2 })
    expect(await telemetry.clearDiagnosticsRuns()).toEqual({ runs: [], removed: 2 })
    expect((await telemetry.fetchDiagnosticsRun("run1")).meta.id).toBe("run1")
  })

  it("surfaces telemetry HTTP failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({}, 503))
    )
    await expect(telemetry.fetchTelemetry()).rejects.toThrow("HTTP 503")
  })
})
