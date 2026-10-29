import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { SafetyLinkStatus } from "@/lib/supervisor"
import { SafetyControl } from "./safety-control"

// Lets React's act() flush updates outside a test renderer.
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: Root
let container: HTMLDivElement

const status = (outputOn: boolean, over: Partial<SafetyLinkStatus> = {}): SafetyLinkStatus => ({
  channel: "can_alm_axol_c",
  connected: true,
  paused: false,
  detected: true,
  reachable: true,
  ageSeconds: 0.1,
  error: null,
  status: {
    estopOk: true,
    latched: outputOn,
    outputOn,
    softwareEnable: outputOn,
    remoteStartActive: false,
    heartbeatOn: false,
    supplyOk: true,
    supplyVolts: 26.4,
    outputVolts: outputOn ? 26 : 0,
    stopCause: outputOn ? "none" : "can",
    firmware: "0.2",
    faults: {
      relay: false,
      k2Stuck: false,
      k2NoClose: false,
      k1PoleB: false,
      remoteStartFailed: false,
      heartbeatLost: false,
    },
  },
  ...over,
})

type Reply = { ok: boolean; status?: number; body: unknown }
let replies: Record<string, Reply>
// While set, status polls wait for it before they answer.
let holdStatus: Promise<void> | null
const calls: string[] = []

beforeEach(() => {
  container = document.createElement("div")
  document.body.appendChild(container)
  root = createRoot(container)
  calls.length = 0
  replies = {}
  holdStatus = null
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? "GET"} ${url}`
      calls.push(key)
      const reply = replies[key] ?? { ok: false, status: 404, body: {} }
      if (key === "GET /api/safety/status" && holdStatus) await holdStatus
      return new Response(JSON.stringify(reply.body), { status: reply.ok ? 200 : reply.status })
    })
  )
})

afterEach(() => {
  vi.useRealTimers()
  act(() => root.unmount())
  container.remove()
  vi.unstubAllGlobals()
})

const flush = async () => {
  for (let i = 0; i < 5; i++) await act(async () => {})
}

const render = async (enabled = true, motorFaults: string[] = []) => {
  await act(async () => root.render(<SafetyControl enabled={enabled} motorFaults={motorFaults} />))
  await flush()
}

const buttonNamed = (text: string) =>
  [...container.querySelectorAll("button")].find((b) => b.textContent?.includes(text))

describe("SafetyControl", () => {
  it("renders nothing without a safety board", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(true, { detected: false }) }
    await render()
    expect(container.textContent).toBe("")
  })

  it("renders nothing and polls nothing while the host is unreachable", async () => {
    await render(false)
    expect(container.textContent).toBe("")
    expect(calls).toEqual([])
  })

  it("cuts power at the first click", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(true) }
    replies["POST /api/safety/stop"] = { ok: true, body: status(false) }
    await render()
    await act(async () => buttonNamed("Cut power")!.click())
    await flush()
    expect(calls).toContain("POST /api/safety/stop")
    expect(buttonNamed("Restore power")).toBeDefined()
  })

  it("keeps the cut state when an older poll answers late", async () => {
    vi.useFakeTimers()
    replies["GET /api/safety/status"] = { ok: true, body: status(true) }
    replies["POST /api/safety/stop"] = { ok: true, body: status(false) }
    await render()
    let release!: () => void
    holdStatus = new Promise((resolve) => (release = resolve))
    await act(async () => vi.advanceTimersByTime(1000))
    await act(async () => buttonNamed("Cut power")!.click())
    await flush()
    expect(buttonNamed("Restore power")).toBeDefined()
    release()
    await flush()
    expect(buttonNamed("Restore power")).toBeDefined()
    expect(buttonNamed("Cut power")).toBeUndefined()
  })

  it("shows an unconfirmed stop in the panel", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(true) }
    replies["POST /api/safety/stop"] = {
      ok: false,
      status: 503,
      body: { error: "The safety board did not confirm the stop. Press the E-Stop." },
    }
    await render()
    await act(async () => buttonNamed("Cut power")!.click())
    await flush()
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("Press the E-Stop")
  })

  it("restores power at the first click", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(false) }
    replies["POST /api/safety/start"] = { ok: true, body: status(true) }
    await render()
    await act(async () => buttonNamed("Restore power")!.click())
    await flush()
    expect(calls).toContain("POST /api/safety/start")
    expect(container.querySelector('[role="dialog"]')).toBeNull()
    expect(buttonNamed("Cut power")).toBeDefined()
  })

  it("shows a refused restore in the panel", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(false) }
    replies["POST /api/safety/start"] = {
      ok: false,
      status: 409,
      body: { error: "Power did not switch on. Check the E-Stop, then try again." },
    }
    await render()
    await act(async () => buttonNamed("Restore power")!.click())
    await flush()
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("try again")
  })

  it("offers no power cycle without a motor fault", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(true) }
    await render()
    expect(buttonNamed("Motor fault")).toBeUndefined()
  })

  it("power-cycles the motors from the motor fault panel", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(true) }
    replies["POST /api/safety/power-cycle"] = { ok: true, body: status(true) }
    await render(true, ["L wrist 2: over current"])
    await act(async () => buttonNamed("Motor fault")!.click())
    expect(container.textContent).toContain("L wrist 2: over current")
    expect(container.textContent).toContain("Support the arms first")
    expect(calls).not.toContain("POST /api/safety/power-cycle")
    await act(async () => buttonNamed("Power-cycle motors")!.click())
    await flush()
    expect(calls).toContain("POST /api/safety/power-cycle")
    expect(container.querySelector('[role="dialog"]')).toBeNull()
  })

  it("keeps Cut power live while a power cycle runs", async () => {
    vi.useFakeTimers()
    replies["GET /api/safety/status"] = { ok: true, body: status(true) }
    let finish!: () => void
    const cycle = new Promise<void>((resolve) => (finish = resolve))
    const fetchMock = vi.mocked(fetch)
    const base = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation(async (url, init) => {
      if (String(url) === "/api/safety/power-cycle") {
        calls.push("POST /api/safety/power-cycle")
        await cycle
        return new Response(JSON.stringify({ error: "Power stays off." }), { status: 409 })
      }
      return base(url, init)
    })
    replies["POST /api/safety/stop"] = { ok: true, body: status(false) }
    await render(true, ["L wrist 2: over current"])
    await act(async () => buttonNamed("Motor fault")!.click())
    await act(async () => buttonNamed("Power-cycle motors")!.click())
    await flush()
    expect(buttonNamed("Power-cycling motors")).toBeDefined()
    // Power is off part way through the cycle: Cut power stays, Restore doesn't.
    replies["GET /api/safety/status"] = { ok: true, body: status(false, { powerCycling: true }) }
    await act(async () => vi.advanceTimersByTime(1000))
    await flush()
    expect(buttonNamed("Restore power")).toBeUndefined()
    const cut = buttonNamed("Cut power")!
    expect(cut.disabled).toBe(false)
    await act(async () => cut.click())
    await flush()
    expect(calls).toContain("POST /api/safety/stop")
    finish()
    await flush()
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("Power stays off.")
  })

  it("shows a refused power cycle in the panel", async () => {
    replies["GET /api/safety/status"] = { ok: true, body: status(true) }
    replies["POST /api/safety/power-cycle"] = {
      ok: false,
      status: 409,
      body: { error: "Stop the running operation before you power-cycle the motors." },
    }
    await render(true, ["R elbow: unreachable"])
    await act(async () => buttonNamed("Motor fault")!.click())
    await act(async () => buttonNamed("Power-cycle motors")!.click())
    await flush()
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "Stop the running operation"
    )
  })

  it("explains why power can't come back on", async () => {
    const pressed = status(false)
    pressed.status = { ...pressed.status!, estopOk: false }
    replies["GET /api/safety/status"] = { ok: true, body: pressed }
    await render()
    expect(buttonNamed("Restore power")).toBeUndefined()
    await act(async () => buttonNamed("E-Stop pressed")!.click())
    expect(container.textContent).toContain("Release the E-Stop first.")
  })
})
