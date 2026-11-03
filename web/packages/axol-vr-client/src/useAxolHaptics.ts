import type { RefObject } from "react"
import { useEffect, useRef } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import {
  HAPTICS_STALE_MS,
  SILENT,
  nextHapticPulse,
  parseHaptics,
  playHapticPulse,
  type AxolHaptics,
  type HapticClock,
} from "./haptics"

/**
 * Plays the teleop server's haptics on the Quest controllers.
 *
 * Listens for `{"type":"haptics"}` pushes on the teleop WebSocket (see
 * `haptics.ts` for the levels and the pattern) and, every XR frame, pulses
 * each hand's controller. Levels that stop arriving for `HAPTICS_STALE_MS`
 * fall silent, so a dropped link never leaves a controller rumbling. An
 * older server that never pushes haptics leaves the controllers alone.
 *
 * A `message` *listener* is added (not `ws.onmessage`) so this coexists with
 * the other consumers on the same socket. Must be used inside a `<Canvas>`
 * (it relies on the R3F frame loop and XR session).
 */
export function useAxolHaptics(wsRef: RefObject<WebSocket | null>): void {
  const { gl } = useThree()

  const levelsRef = useRef<AxolHaptics | null>(null)
  const receivedAtRef = useRef(0)
  const clocksRef = useRef<Record<"left" | "right", HapticClock>>({
    left: { nextAt: 0 },
    right: { nextAt: 0 },
  })
  // WebSocket we've attached the haptics listener to, to avoid re-attaching.
  const attachedWsRef = useRef<WebSocket | null>(null)
  const listenerRef = useRef<((event: MessageEvent) => void) | null>(null)

  useEffect(
    () => () => {
      if (attachedWsRef.current && listenerRef.current) {
        attachedWsRef.current.removeEventListener("message", listenerRef.current)
      }
      attachedWsRef.current = null
      listenerRef.current = null
    },
    []
  )

  useFrame(() => {
    const ws = wsRef.current
    if (ws !== attachedWsRef.current) {
      if (attachedWsRef.current && listenerRef.current) {
        attachedWsRef.current.removeEventListener("message", listenerRef.current)
      }
      attachedWsRef.current = ws
      listenerRef.current = null
      levelsRef.current = null
      if (ws) {
        const onMessage = (event: MessageEvent) => {
          if (wsRef.current !== ws) return
          try {
            const msg = JSON.parse(event.data as string) as { type: string; value: unknown }
            if (msg.type !== "haptics") return
            const levels = parseHaptics(msg.value)
            if (!levels) return
            levelsRef.current = levels
            receivedAtRef.current = performance.now()
          } catch {
            // ignore malformed messages
          }
        }
        listenerRef.current = onMessage
        ws.addEventListener("message", onMessage)
      }
    }

    const session = gl.xr.getSession()
    if (!session) return
    const now = performance.now()
    const levels = now - receivedAtRef.current > HAPTICS_STALE_MS ? null : levelsRef.current
    for (const source of session.inputSources) {
      const hand = source.handedness
      if (hand !== "left" && hand !== "right") continue
      const pulse = nextHapticPulse(levels?.[hand] ?? SILENT, clocksRef.current[hand], now)
      if (pulse) playHapticPulse(source.gamepad, pulse)
    }
  })
}
