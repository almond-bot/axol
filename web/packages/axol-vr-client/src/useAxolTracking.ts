import type { RefObject } from "react"
import { useRef } from "react"
import { useFrame, useThree } from "@react-three/fiber"

/**
 * Tracks whether the robot is currently engaged (mirroring the operator), as a
 * frame-readable ref.
 *
 * The server owns the engage toggle — grips, X/reset, and saving all flip it
 * there — and broadcasts it over the teleop WebSocket as
 * `{"type":"tracking", "value": boolean}`. That push is the source of truth, so
 * this stays correct where a purely client-side mirror would drift.
 *
 * Until the server has pushed tracking state at least once, this falls back to
 * mirroring the grip toggle locally (both grips together engage, either grip
 * alone disengages — the same edges as the teleop server), so callers behave
 * correctly against an older backend that doesn't broadcast tracking yet.
 *
 * A `message` *listener* is added (not `ws.onmessage`) so this coexists with
 * other consumers on the same socket (e.g. `AxolVRClient`, `useAxolVideo`).
 *
 * Returns a ref rather than state because it is meant to be read inside a
 * per-frame loop without triggering React re-renders. Must be used inside a
 * `<Canvas>` (it relies on the R3F frame loop and XR session).
 */
export function useAxolTracking(wsRef: RefObject<WebSocket | null>): RefObject<boolean> {
  const { gl } = useThree()

  // Whether the robot is engaged, as reported by the server's tracking pushes.
  const engagedRef = useRef(false)
  // True once a tracking push has arrived on this connection; until then we
  // fall back to the local grip mirror below.
  const serverSeenRef = useRef(false)
  // Local fallback mirror of the engage toggle plus its edge-detection state.
  const mirrorEngagedRef = useRef(false)
  const prevBothGripsRef = useRef(false)
  const prevEitherGripRef = useRef(false)
  // WebSocket we've attached the tracking listener to, to avoid re-attaching.
  const attachedWsRef = useRef<WebSocket | null>(null)

  useFrame(() => {
    // Attach the tracking listener whenever the socket changes (added as an
    // extra listener so other consumers' handlers keep working), and reset the
    // engage state for the fresh connection.
    const ws = wsRef.current
    if (ws !== attachedWsRef.current) {
      attachedWsRef.current = ws
      engagedRef.current = false
      serverSeenRef.current = false
      mirrorEngagedRef.current = false
      prevBothGripsRef.current = false
      prevEitherGripRef.current = false
      ws?.addEventListener("message", (event: MessageEvent) => {
        try {
          const msg = JSON.parse(event.data as string) as { type: string; value: unknown }
          if (msg.type === "tracking") {
            serverSeenRef.current = true
            engagedRef.current = !!msg.value
          }
        } catch {
          // ignore malformed messages
        }
      })
    }

    const session = gl.xr.getSession()
    const sources = session ? Array.from(session.inputSources) : []
    const gripPressed = (hand: "left" | "right") => {
      const src = sources.find((s) => s.handedness === hand)
      return (src?.gamepad?.buttons?.[1]?.value ?? 0) >= 1.0
    }
    const bothGrips = gripPressed("left") && gripPressed("right")
    const eitherGrip = gripPressed("left") || gripPressed("right")
    if (!mirrorEngagedRef.current) {
      if (bothGrips && !prevBothGripsRef.current) mirrorEngagedRef.current = true
    } else {
      if (eitherGrip && !prevEitherGripRef.current) mirrorEngagedRef.current = false
    }
    prevBothGripsRef.current = bothGrips
    prevEitherGripRef.current = eitherGrip

    // The server's push wins once seen; until then, follow the local mirror.
    if (!serverSeenRef.current) {
      engagedRef.current = mirrorEngagedRef.current
    }
  })

  return engagedRef
}
