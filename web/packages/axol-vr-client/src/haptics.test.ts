import { describe, expect, it, vi } from "vitest"

import {
  THROB_FAST_MS,
  THROB_SLOW_MS,
  TICK_FAST_MS,
  TICK_SLOW_MS,
  nextHapticPulse,
  parseHaptics,
  playHapticPulse,
  type HapticClock,
} from "./haptics"

describe("parseHaptics", () => {
  it("clamps levels and fills a missing hand with silence", () => {
    expect(parseHaptics({ left: { near: 1.5, force: -2 } })).toEqual({
      left: { near: 1, force: 0 },
      right: { near: 0, force: 0 },
    })
  })

  it("rejects a value that is not an object", () => {
    expect(parseHaptics(null)).toBeNull()
    expect(parseHaptics("on")).toBeNull()
  })
})

describe("nextHapticPulse", () => {
  it("stays silent at zero", () => {
    const clock: HapticClock = { nextAt: 0 }
    expect(nextHapticPulse({ near: 0, force: 0 }, clock, 1000)).toBeNull()
  })

  it("ticks faster as the arm gets closer", () => {
    const far: HapticClock = { nextAt: 0 }
    const close: HapticClock = { nextAt: 0 }
    const a = nextHapticPulse({ near: 0.1, force: 0 }, far, 1000)
    const b = nextHapticPulse({ near: 1, force: 0 }, close, 1000)
    expect(a).not.toBeNull()
    expect(b!.intensity).toBeGreaterThan(a!.intensity)
    expect(far.nextAt - 1000).toBeGreaterThan(close.nextAt - 1000)
    expect(close.nextAt - 1000).toBeCloseTo(TICK_FAST_MS)
    expect(far.nextAt - 1000).toBeLessThanOrEqual(TICK_SLOW_MS)
    // No second tick before the interval is up.
    expect(nextHapticPulse({ near: 1, force: 0 }, close, 1000 + TICK_FAST_MS / 2)).toBeNull()
    expect(nextHapticPulse({ near: 1, force: 0 }, close, 1000 + TICK_FAST_MS)).not.toBeNull()
  })

  it("throbs harder and faster as the push grows, and wins over the ticks", () => {
    const light: HapticClock = { nextAt: 0 }
    const hard: HapticClock = { nextAt: 0 }
    const a = nextHapticPulse({ near: 1, force: 0.1 }, light, 1000)!
    const b = nextHapticPulse({ near: 0, force: 0.7 }, hard, 1000)!
    expect(b.intensity).toBeGreaterThan(a.intensity)
    expect(hard.nextAt).toBeLessThan(light.nextAt)
    expect(light.nextAt - 1000).toBeLessThanOrEqual(THROB_SLOW_MS)
    // Light throbs leave gaps; the ticks' near level doesn't speed them up.
    expect(light.nextAt - 1000).toBeGreaterThan(a.duration)
    expect(a.duration).toBeGreaterThan(30)
  })

  it("merges into a solid rumble at full force", () => {
    const clock: HapticClock = { nextAt: 0 }
    const full = nextHapticPulse({ near: 0, force: 1 }, clock, 0)!
    expect(full.intensity).toBe(1)
    expect(clock.nextAt).toBeCloseTo(THROB_FAST_MS)
    expect(full.duration).toBeGreaterThan(THROB_FAST_MS)
    expect(nextHapticPulse({ near: 0, force: 1 }, clock, THROB_FAST_MS)).not.toBeNull()
  })

  it("ticks at once on the next approach after going quiet", () => {
    const clock: HapticClock = { nextAt: 0 }
    nextHapticPulse({ near: 0.2, force: 0 }, clock, 1000)
    nextHapticPulse({ near: 0, force: 0 }, clock, 1010)
    expect(nextHapticPulse({ near: 0.2, force: 0 }, clock, 1020)).not.toBeNull()
  })
})

describe("playHapticPulse", () => {
  it("uses the WebXR pulse actuator", () => {
    const pulse = vi.fn(() => Promise.resolve(true))
    const gamepad = { hapticActuators: [{ pulse }] } as unknown as Gamepad
    expect(playHapticPulse(gamepad, { intensity: 0.5, duration: 20 })).toBe(true)
    expect(pulse).toHaveBeenCalledWith(0.5, 20)
  })

  it("falls back to the vibration actuator", () => {
    const playEffect = vi.fn(() => Promise.resolve("complete"))
    const gamepad = { vibrationActuator: { playEffect } } as unknown as Gamepad
    expect(playHapticPulse(gamepad, { intensity: 0.5, duration: 20 })).toBe(true)
    expect(playEffect).toHaveBeenCalledWith("dual-rumble", {
      duration: 20,
      strongMagnitude: 0.5,
      weakMagnitude: 0.5,
    })
  })

  it("reports a controller without an actuator", () => {
    expect(playHapticPulse({} as Gamepad, { intensity: 0.5, duration: 20 })).toBe(false)
    expect(playHapticPulse(null, { intensity: 0.5, duration: 20 })).toBe(false)
  })
})
