import type { RefObject } from "react"
import { useEffect, useRef } from "react"
import { useFrame } from "@react-three/fiber"

/** Base transform of the virtual robot in the XR reference space. */
export interface AxolUrdfBase {
  /** Position in metres. */
  pos: [number, number, number]
  /** Orientation quaternion, xyzw. */
  quat: [number, number, number, number]
}

/**
 * Latest URDF overlay state pushed by the server in absolute (Mantis) mode.
 *
 * `base` is the robot base transform solved at engage (null before the first
 * engage — the overlay should stay hidden until then). `joints` is the current
 * IK solution keyed by URDF joint name.
 */
export interface AxolUrdfState {
  base: AxolUrdfBase | null
  joints: Record<string, number>
  engaged: boolean
  /** False when the producer and viewer use unregistered tracking worlds. */
  viewerWorldAligned: boolean
}

/**
 * Tracks the server's `{"type":"urdf_state"}` pushes as a frame-readable ref.
 *
 * In absolute (Mantis) mode the teleop server streams the engage-calibrated robot
 * base transform plus the live IK joint solution (~30 Hz) so the headset can
 * render the virtual robot exactly where the calibration placed it — the
 * operator verifies hardware↔URDF alignment by checking the virtual gripper
 * sits on the physical one.
 *
 * A `message` *listener* is added (not `ws.onmessage`) so this coexists with
 * other consumers on the same socket. Returns a ref rather than state because
 * it is read inside a per-frame loop without triggering React re-renders. Must
 * be used inside a `<Canvas>`.
 */
export function useAxolUrdfState(
  wsRef: RefObject<WebSocket | null>
): RefObject<AxolUrdfState | null> {
  const stateRef = useRef<AxolUrdfState | null>(null)
  // WebSocket we've attached the listener to, to avoid re-attaching.
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
      stateRef.current = null
      if (!ws) return
      const onMessage = (event: MessageEvent) => {
        // A message already queued by a superseded socket must not repopulate
        // the state that was just cleared for the new connection.
        if (wsRef.current !== ws) return
        try {
          const msg = JSON.parse(event.data as string) as {
            type?: string
            viewer_world_aligned?: boolean
          } & Partial<AxolUrdfState>
          if (msg.type === "urdf_state") {
            stateRef.current = {
              base: msg.base ?? null,
              joints: msg.joints ?? {},
              engaged: !!msg.engaged,
              viewerWorldAligned: msg.viewer_world_aligned !== false,
            }
          }
        } catch {
          // ignore malformed messages
        }
      }
      listenerRef.current = onMessage
      ws.addEventListener("message", onMessage)
    }
  })

  return stateRef
}
