import { beforeEach, describe, expect, it } from "vitest"

import {
  DEFAULT_CAMERAS,
  LEGACY_CAMERA_STORAGE,
  cameraStorageKey,
  loadCachedCameras,
  persistCachedCameras,
} from "./camera-cache"
import type { CameraSpec } from "./supervisor"

const mine: CameraSpec = {
  ...DEFAULT_CAMERAS,
  serials: { overhead: "111", left_arm: "222", right_arm: "333" },
}

describe("per-host camera cache", () => {
  beforeEach(() => localStorage.clear())

  it("keeps each host's spec apart", () => {
    persistCachedCameras("10.0.0.5", mine)
    expect(loadCachedCameras("10.0.0.5").serials.overhead).toBe("111")
    expect(loadCachedCameras("customer.local")).toEqual(DEFAULT_CAMERAS)
  })

  it("treats spellings of one host origin as the same host", () => {
    expect(cameraStorageKey("10.0.0.5")).toBe(cameraStorageKey("https://10.0.0.5:8001"))
    expect(cameraStorageKey("10.0.0.5")).not.toBe(cameraStorageKey("10.0.0.6"))
  })

  it("ignores the browser-wide legacy spec unless asked for it", () => {
    localStorage.setItem(LEGACY_CAMERA_STORAGE, JSON.stringify(mine))
    expect(loadCachedCameras("customer.local")).toEqual(DEFAULT_CAMERAS)
    expect(loadCachedCameras("customer.local", { legacy: true }).serials.left_arm).toBe("222")
  })

  it("prefers the host's own copy over the legacy spec", () => {
    localStorage.setItem(LEGACY_CAMERA_STORAGE, JSON.stringify(mine))
    persistCachedCameras("old.local", {
      ...DEFAULT_CAMERAS,
      serials: { overhead: "999", left_arm: "", right_arm: "" },
    })
    expect(loadCachedCameras("old.local", { legacy: true }).serials.overhead).toBe("999")
  })

  it("migrates older stored field shapes", () => {
    persistCachedCameras("old.local", {
      serials: { overhead: "1" },
      resolution: "HD1080",
      stream_eyes: { overhead: "left" },
    } as unknown as CameraSpec)
    const spec = loadCachedCameras("old.local")
    expect(spec.stream_resolution).toBe("HD1080")
    expect(spec.serials).toEqual({ overhead: "1", left_arm: "", right_arm: "" })
    expect(spec.stream).toEqual({ overhead: "left" })
  })

  it("falls back to defaults on malformed storage", () => {
    localStorage.setItem(cameraStorageKey("bad.local"), "{not json")
    expect(loadCachedCameras("bad.local")).toEqual(DEFAULT_CAMERAS)
  })
})
