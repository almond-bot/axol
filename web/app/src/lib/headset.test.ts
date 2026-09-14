import { afterEach, describe, expect, it } from "vitest"

import { isHeadsetBrowser } from "./headset"

const originalUserAgent = navigator.userAgent

afterEach(() => {
  Object.defineProperty(navigator, "userAgent", { configurable: true, value: originalUserAgent })
})

describe("headset detection", () => {
  it.each(["OculusBrowser/32", "Mozilla Quest 3", "PicoBrowser VR"])("recognizes %s", (ua) => {
    Object.defineProperty(navigator, "userAgent", { configurable: true, value: ua })
    expect(isHeadsetBrowser()).toBe(true)
  })

  it("does not classify a desktop browser as a headset", () => {
    Object.defineProperty(navigator, "userAgent", { configurable: true, value: "Mozilla Chrome" })
    expect(isHeadsetBrowser()).toBe(false)
  })
})
