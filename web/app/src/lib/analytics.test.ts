import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const init = vi.fn()
const capture = vi.fn()
const group = vi.fn()
const resetGroups = vi.fn()
vi.mock("posthog-js", () => ({
  default: { init, capture, group, resetGroups },
}))

type Analytics = typeof import("./analytics")

async function load(): Promise<Analytics> {
  vi.resetModules()
  return import("./analytics")
}

const info = {
  hostname: "axol-7",
  lanIp: "192.168.1.50",
  viewerPort: 8002,
  vrPort: 8000,
  version: "0.2.31",
  commit: "abc123",
  releaseInstall: true,
}

beforeEach(() => {
  init.mockReset()
  capture.mockReset()
  group.mockReset()
  resetGroups.mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("analyticsConfig", () => {
  it("stays off without a key", async () => {
    const { analyticsConfig } = await load()
    expect(analyticsConfig({})).toBeNull()
    expect(analyticsConfig({ VITE_POSTHOG_KEY: "  " })).toBeNull()
  })

  it("stays off in a dev build even with a key", async () => {
    const { analyticsConfig } = await load()
    expect(analyticsConfig({ VITE_POSTHOG_KEY: "phc_x", DEV: true })).toBeNull()
  })

  it("defaults the host to PostHog's US cloud", async () => {
    const { analyticsConfig, DEFAULT_ANALYTICS_HOST } = await load()
    expect(analyticsConfig({ VITE_POSTHOG_KEY: "phc_x" })).toEqual({
      key: "phc_x",
      host: DEFAULT_ANALYTICS_HOST,
    })
    expect(
      analyticsConfig({ VITE_POSTHOG_KEY: "phc_x", VITE_POSTHOG_HOST: "https://eu.i.posthog.com" })
    ).toEqual({ key: "phc_x", host: "https://eu.i.posthog.com" })
  })
})

describe("initAnalytics", () => {
  it("does nothing without a key, and every call stays a no-op", async () => {
    const a = await load()
    expect(a.initAnalytics({})).toBe(false)
    expect(a.analyticsEnabled()).toBe(false)
    a.track("robot_disconnected", {})
    a.identifyRobot(info)
    a.forgetRobot()
    expect(init).not.toHaveBeenCalled()
    expect(capture).not.toHaveBeenCalled()
    expect(group).not.toHaveBeenCalled()
    expect(resetGroups).not.toHaveBeenCalled()
  })

  it("starts PostHog with explicit events only and masked replay", async () => {
    const a = await load()
    expect(a.initAnalytics({ VITE_POSTHOG_KEY: "phc_x" })).toBe(true)
    expect(a.analyticsEnabled()).toBe(true)
    expect(init).toHaveBeenCalledWith(
      "phc_x",
      expect.objectContaining({
        api_host: a.DEFAULT_ANALYTICS_HOST,
        capture_pageview: "history_change",
        autocapture: false,
        capture_exceptions: true,
        session_recording: { maskAllInputs: true },
      })
    )
  })

  it("stays off when PostHog itself throws", async () => {
    init.mockImplementation(() => {
      throw new Error("blocked")
    })
    const a = await load()
    expect(a.initAnalytics({ VITE_POSTHOG_KEY: "phc_x" })).toBe(false)
    a.track("robot_disconnected", {})
    expect(capture).not.toHaveBeenCalled()
  })
})

describe("with analytics on", () => {
  let a: Analytics
  beforeEach(async () => {
    a = await load()
    a.initAnalytics({ VITE_POSTHOG_KEY: "phc_x" })
  })

  it("sends events with their properties", () => {
    a.track("operation_started", { operation: "teleop", sim: true, device: "axol" })
    expect(capture).toHaveBeenCalledWith("operation_started", {
      operation: "teleop",
      sim: true,
      device: "axol",
    })
  })

  it("files the browser under the robot as a group", () => {
    a.identifyRobot(info)
    expect(group).toHaveBeenCalledWith("robot", "axol-7", {
      name: "axol-7",
      version: "0.2.31",
      commit: "abc123",
      release_install: true,
    })
    a.identifyRobot({ hostname: "bench", lanIp: "x", viewerPort: 1, vrPort: 2 })
    expect(group).toHaveBeenLastCalledWith("robot", "bench", {
      name: "bench",
      version: null,
      commit: null,
      release_install: null,
    })
    a.forgetRobot()
    expect(resetGroups).toHaveBeenCalled()
  })

  it("never throws out of a failing capture", () => {
    capture.mockImplementation(() => {
      throw new Error("offline")
    })
    expect(() => a.track("robot_disconnected", {})).not.toThrow()
  })
})

describe("checkoutProperties", () => {
  it("mirrors the site's checkout events, with the value in dollars", async () => {
    const { checkoutProperties } = await load()
    expect(
      checkoutProperties(
        { service: "dataset", method: "your_axol", hours: 12, operation: "collect-data" },
        { unitAmount: 9900, currency: "usd" }
      )
    ).toEqual({
      service: "dataset",
      collection_method: "your_axol",
      hours: 12,
      operation: "collect-data",
      value: 1188,
      currency: "USD",
    })
  })

  it("leaves the value out when the price never arrived", async () => {
    const { checkoutProperties } = await load()
    expect(
      checkoutProperties({ service: "task", method: null, hours: 2, operation: "overview" }, null)
    ).toMatchObject({ collection_method: null, value: null, currency: null })
  })
})

describe("versionMismatchProperties", () => {
  it("flattens the mismatch into snake_case properties", async () => {
    const { versionMismatchProperties } = await load()
    expect(
      versionMismatchProperties({
        local: false,
        serverDev: false,
        serverOlder: true,
        uiCommit: "u",
        serverCommit: "s",
        uiVersion: "1.2.0",
        serverVersion: "1.1.9",
      })
    ).toEqual({
      local: false,
      server_dev: false,
      server_older: true,
      ui_version: "1.2.0",
      server_version: "1.1.9",
    })
  })
})
