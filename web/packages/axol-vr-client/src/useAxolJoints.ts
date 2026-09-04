import type { RefObject } from "react"
import { useEffect, useRef } from "react"
import type { AxolJointState } from "./types"

/** Latest joint push plus the local time (ms, `performance.now()`) it arrived. */
export type AxolJointSample = AxolJointState & { receivedAt: number }

/**
 * Mirrors the teleop server's live joint state as a frame-readable ref.
 *
 * The server pushes `{"type":"joints","value":{q, l_grip, r_grip, engaged}}`
 * ~20x/s over the teleop WebSocket (see `AxolJointState`); this hook keeps the
 * newest sample so a per-frame loop (the ghost robot overlay) can pose a
 * model without React re-renders. `null` until the first push on the current
 * connection — an older server that never pushes joints simply leaves it null.
 *
 * A `message` *listener* is added (not `ws.onmessage`) so this coexists with
 * the other consumers on the same socket.
 */
export function useAxolJoints(
  wsRef: RefObject<WebSocket | null>,
  connected: boolean
): RefObject<AxolJointSample | null> {
  const sampleRef = useRef<AxolJointSample | null>(null)

  useEffect(() => {
    const ws = wsRef.current
    sampleRef.current = null
    if (!connected || !ws) return
    const onMessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as { type: string; value: unknown }
        if (msg.type !== "joints") return
        const v = msg.value as Partial<AxolJointState> | null
        if (!v || typeof v.q !== "object" || v.q === null) return
        const pair = v.pair
        sampleRef.current = {
          q: v.q,
          l_grip: typeof v.l_grip === "number" ? v.l_grip : 1,
          r_grip: typeof v.r_grip === "number" ? v.r_grip : 1,
          engaged: !!v.engaged,
          pair:
            pair && typeof pair === "object" && typeof pair.width === "number"
              ? {
                  aligned: !!pair.aligned,
                  width: pair.width,
                  tilt: typeof pair.tilt === "number" ? pair.tilt : 0,
                }
              : null,
          receivedAt: performance.now(),
        }
      } catch {
        // ignore malformed messages
      }
    }
    ws.addEventListener("message", onMessage)
    return () => {
      ws.removeEventListener("message", onMessage)
      sampleRef.current = null
    }
  }, [wsRef, connected])

  return sampleRef
}
