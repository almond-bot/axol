import type { RefObject } from "react"
import { useRef } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import { AxolState } from "./types"

const L_ELBOW_JOINT = "left-arm-lower" as XRBodyJoint
const R_ELBOW_JOINT = "right-arm-lower" as XRBodyJoint

export function AxolVRClient({
  wsRef,
  onStateChange,
  onPendingRecording,
  onExit,
}: {
  wsRef: RefObject<WebSocket | null>
  onStateChange?: (state: AxolState) => void
  onPendingRecording?: (pendingAt: number | null) => void
  onExit?: () => void
}) {
  const { gl } = useThree()

  const stateRef = useRef<AxolState>(AxolState.Teleop)
  const seqRef = useRef(0)
  const prevXRef = useRef(false)
  const prevYRef = useRef(false)
  const prevARef = useRef(false)
  const prevBRef = useRef(false)
  const recordingPendingAtRef = useRef<number | null>(null)
  // Server-pushed state override (e.g. "saving"). Applied at start of each frame.
  const serverStateRef = useRef<AxolState | null>(null)
  // Track which WebSocket we have attached onmessage to avoid re-attaching.
  const wsWithHandlerRef = useRef<WebSocket | null>(null)

  useFrame(() => {
    // Attach onmessage to the WebSocket whenever it changes so we can receive
    // server-pushed state overrides (e.g. the "saving" state after recording).
    const currentWs = wsRef.current
    if (currentWs !== wsWithHandlerRef.current) {
      wsWithHandlerRef.current = currentWs
      if (currentWs) {
        currentWs.onmessage = (event: MessageEvent) => {
          try {
            const msg = JSON.parse(event.data as string) as { type: string; value: string }
            if (msg.type === "state") {
              serverStateRef.current = msg.value as AxolState
            }
          } catch {
            // ignore malformed messages
          }
        }
      }
    }

    const session = gl.xr.getSession()
    if (!session) return

    const frame = gl.xr.getFrame()
    const refSpace = gl.xr.getReferenceSpace()
    if (!frame || !refSpace) return

    const leftSource = Array.from(session.inputSources).find(
      (s: XRInputSource) => s.handedness === "left"
    )
    const rightSource = Array.from(session.inputSources).find(
      (s: XRInputSource) => s.handedness === "right"
    )

    const xPressed = leftSource?.gamepad?.buttons[4]?.pressed ?? false
    const yPressed = leftSource?.gamepad?.buttons[5]?.pressed ?? false
    const aPressed = rightSource?.gamepad?.buttons[4]?.pressed ?? false
    const bPressed = rightSource?.gamepad?.buttons[5]?.pressed ?? false

    const xEdge = xPressed && !prevXRef.current
    const yEdge = yPressed && !prevYRef.current
    const aEdge = aPressed && !prevARef.current
    const bEdge = bPressed && !prevBRef.current

    prevXRef.current = xPressed
    prevYRef.current = yPressed
    prevARef.current = aPressed
    prevBRef.current = bPressed

    // Apply server-pushed state override before processing button presses.
    if (serverStateRef.current !== null) {
      const next = serverStateRef.current
      serverStateRef.current = null
      stateRef.current = next
      // Cancel any pending countdown when entering saving state.
      if (next === AxolState.Saving) {
        recordingPendingAtRef.current = null
        onPendingRecording?.(null)
      }
      onStateChange?.(next)
    }

    const state = stateRef.current
    const isSaving = state === AxolState.Saving

    function setState(next: AxolState) {
      stateRef.current = next
      onStateChange?.(next)
    }

    const isPending = recordingPendingAtRef.current !== null

    // X — reset; also cancels pending/recording (not allowed while saving)
    let reset = false
    if (xEdge && !isSaving) {
      reset = true
      if (state === AxolState.Recording || isPending) {
        setState(AxolState.DataCollection)
        recordingPendingAtRef.current = null
        onPendingRecording?.(null)
      }
    }

    // Y (left) — exit XR
    if (yEdge) {
      onExit?.()
    }

    // B (right) — swap teleop ↔ data_collection (disabled when recording, pending, or saving)
    if (bEdge && state !== AxolState.Recording && !isPending && !isSaving) {
      setState(state === AxolState.Teleop ? AxolState.DataCollection : AxolState.Teleop)
    }

    // A — start pending (3s countdown) or stop recording; blocked while saving
    if (aEdge && !isSaving) {
      if (state === AxolState.Recording) {
        setState(AxolState.DataCollection)
      } else if (state === AxolState.DataCollection && !isPending) {
        recordingPendingAtRef.current = Date.now()
        onPendingRecording?.(recordingPendingAtRef.current)
      } else if (isPending) {
        recordingPendingAtRef.current = null
        onPendingRecording?.(null)
      }
    }

    // Promote pending → recording after 3s
    if (
      recordingPendingAtRef.current !== null &&
      Date.now() - recordingPendingAtRef.current >= 3000
    ) {
      setState(AxolState.Recording)
      recordingPendingAtRef.current = null
      onPendingRecording?.(null)
    }

    const dc = wsRef.current
    if (!dc || dc.readyState !== WebSocket.OPEN) return

    function getPose(space: XRSpace | null | undefined) {
      if (!space) return null
      const pose = frame.getPose(space, refSpace!)
      if (!pose) return null
      const { position: p, orientation: o } = pose.transform
      return {
        position: { x: p.x, y: p.y, z: p.z },
        quaternion: { x: o.x, y: o.y, z: o.z, w: o.w },
      }
    }

    function getPosition(space: XRSpace | null | undefined) {
      if (!space) return null
      const pose = frame.getPose(space, refSpace!)
      if (!pose) return null
      const { position: p } = pose.transform
      return { x: p.x, y: p.y, z: p.z }
    }

    const l_ee = getPose(leftSource?.targetRaySpace)
    const r_ee = getPose(rightSource?.targetRaySpace)

    if (!l_ee || !r_ee) return

    const body = (frame as XRFrame & { body?: XRBody }).body
    const l_elbow = getPosition(body?.get(L_ELBOW_JOINT))
    const r_elbow = getPosition(body?.get(R_ELBOW_JOINT))

    if (!l_elbow || !r_elbow) return

    const l_grip = 1 - (leftSource?.gamepad?.buttons[0]?.value ?? 0)
    const r_grip = 1 - (rightSource?.gamepad?.buttons[0]?.value ?? 0)
    const l_lock = (leftSource?.gamepad?.buttons[1]?.value ?? 0) >= 1.0
    const r_lock = (rightSource?.gamepad?.buttons[1]?.value ?? 0) >= 1.0

    dc.send(
      JSON.stringify({
        l_ee,
        r_ee,
        l_elbow,
        r_elbow,
        l_lock,
        r_lock,
        l_grip,
        r_grip,
        reset,
        state: stateRef.current,
        seq: ++seqRef.current,
      })
    )
  })

  return null
}
