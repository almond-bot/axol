/**
 * Controller haptics: turns the server's per-hand levels into vibration pulses.
 *
 * The teleop server pushes `{"type":"haptics","value":{"left":{near, force},
 * "right":{near, force}}}` (~30x/s while anything plays, then one all-zero
 * push). Both levels are 0 to 1:
 *
 * - `near`: how close that arm is to the IK's arm/body collision boundary.
 *   Played as short ticks that speed up as it rises, like a parking sensor.
 * - `force`: how hard that arm pushes on something (torque past nominal).
 *   Played as a continuous rumble that grows with it, and wins over the ticks.
 *
 * Pure so the pattern is testable; `useAxolHaptics` drives the controllers.
 */

/** One hand's levels, each 0 to 1. */
export type AxolHandHaptics = { near: number; force: number }

/** Both hands' levels, as the server pushes them. */
export type AxolHaptics = { left: AxolHandHaptics; right: AxolHandHaptics }

/** One pulse to play: strength 0 to 1 and length in ms. */
export type HapticPulse = { intensity: number; duration: number }

/** Per-hand pattern state (when the next pulse is due, `performance.now()` ms). */
export type HapticClock = { nextAt: number }

export const SILENT: AxolHandHaptics = { near: 0, force: 0 }

// Proximity ticks: interval from just inside the onset down to the boundary,
// and tick strength over the same range.
export const TICK_SLOW_MS = 350
export const TICK_FAST_MS = 70
const TICK_MS = 25
const TICK_MIN = 0.2
const TICK_MAX = 0.6
// Push rumble: re-issued before the previous pulse ends so it stays solid.
export const RUMBLE_EVERY_MS = 40
const RUMBLE_MS = 60
const RUMBLE_MIN = 0.3
// Levels older than this are dropped (link lost mid-rumble).
export const HAPTICS_STALE_MS = 300

function level(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : 0
}

function parseHand(raw: unknown): AxolHandHaptics {
  if (!raw || typeof raw !== "object") return SILENT
  const hand = raw as Record<string, unknown>
  return { near: level(hand.near), force: level(hand.force) }
}

/** Parse a `haptics` push value; null when it isn't one. */
export function parseHaptics(value: unknown): AxolHaptics | null {
  if (!value || typeof value !== "object") return null
  const v = value as Record<string, unknown>
  return { left: parseHand(v.left), right: parseHand(v.right) }
}

/**
 * The pulse to start now for one hand, or null to leave the controller alone.
 *
 * Call once per XR frame; `clock` carries the pattern between calls.
 */
export function nextHapticPulse(
  hand: AxolHandHaptics,
  clock: HapticClock,
  now: number
): HapticPulse | null {
  if (hand.force > 0) {
    if (now < clock.nextAt) return null
    clock.nextAt = now + RUMBLE_EVERY_MS
    return { intensity: RUMBLE_MIN + (1 - RUMBLE_MIN) * hand.force, duration: RUMBLE_MS }
  }
  if (hand.near > 0) {
    if (now < clock.nextAt) return null
    clock.nextAt = now + TICK_SLOW_MS + (TICK_FAST_MS - TICK_SLOW_MS) * hand.near
    return { intensity: TICK_MIN + (TICK_MAX - TICK_MIN) * hand.near, duration: TICK_MS }
  }
  // Silent: the next approach ticks at once.
  clock.nextAt = 0
  return null
}

// WebXR controllers expose `hapticActuators[].pulse` (Quest Browser); the
// standard Gamepad `vibrationActuator` is the fallback. Neither `pulse` nor
// `hapticActuators` is in TypeScript's DOM types.
type PulseActuator = { pulse?: (intensity: number, duration: number) => unknown }
type HapticGamepad = Gamepad & { hapticActuators?: readonly PulseActuator[] }

/** Play `pulse` on a controller's gamepad. Returns false when it has no actuator. */
export function playHapticPulse(gamepad: Gamepad | null | undefined, pulse: HapticPulse): boolean {
  if (!gamepad) return false
  const actuator = (gamepad as HapticGamepad).hapticActuators?.[0]
  if (typeof actuator?.pulse === "function") {
    void Promise.resolve(actuator.pulse(pulse.intensity, pulse.duration)).catch(() => {})
    return true
  }
  const vibration = gamepad.vibrationActuator
  if (vibration && typeof vibration.playEffect === "function") {
    void vibration
      .playEffect("dual-rumble", {
        duration: pulse.duration,
        strongMagnitude: pulse.intensity,
        weakMagnitude: pulse.intensity,
      })
      .catch(() => {})
    return true
  }
  return false
}
