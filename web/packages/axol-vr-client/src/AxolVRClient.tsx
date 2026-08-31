import type { RefObject } from "react"
import { useRef } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import { AxolState } from "./types"
import type { AxolMode, ConfirmAction } from "./types"

const L_ELBOW_JOINT = "left-arm-lower" as XRBodyJoint
const R_ELBOW_JOINT = "right-arm-lower" as XRBodyJoint

// Pose sinks (WebSocket and RTCDataChannel) both expose `.send(string)`; this is
// the minimal shape AxolVRClient needs to ship a frame.
type PoseSink = { send: (data: string) => void }

/** Best open network pose transport: the low-latency WebRTC data channel when
 *  open (UDP; right path over a relayed Funnel, equal-or-better on a LAN),
 *  otherwise the main WebSocket. Null when neither is up. */
function pickNetworkSink(
  poseChannelRef: RefObject<RTCDataChannel | null> | undefined,
  wsRef: RefObject<WebSocket | null>
): PoseSink | null {
  const channel = poseChannelRef?.current
  if (channel && channel.readyState === "open") return channel
  const ws = wsRef.current
  return ws && ws.readyState === WebSocket.OPEN ? ws : null
}

/** USB pose transport (the Quest-over-USB `adb reverse` WebSocket) when open. */
function pickUsbSink(poseWsRef: RefObject<WebSocket | null> | undefined): PoseSink | null {
  const ws = poseWsRef?.current
  return ws && ws.readyState === WebSocket.OPEN ? ws : null
}

export function AxolVRClient({
  wsRef,
  poseWsRef,
  poseChannelRef,
  onStateChange,
  onPendingRecording,
  onPendingConfirm,
  onMode,
  onEpisode,
  onExit,
}: {
  wsRef: RefObject<WebSocket | null>
  // Dedicated pose WebSocket — the Quest-over-USB `adb reverse` tunnel. When
  // present and open, each frame is sent over BOTH this and the best network
  // transport below: the server de-dupes by seq and prefers the low-latency
  // USB stream, falling back to the network frames the instant USB goes quiet,
  // so a USB drop fails over with no reconnect.
  poseWsRef?: RefObject<WebSocket | null>
  // Low-latency WebRTC pose data channel (see useAxolControlChannel). Preferred
  // network transport when open; the main WebSocket is the network fallback.
  poseChannelRef?: RefObject<RTCDataChannel | null>
  onStateChange?: (state: AxolState) => void
  onPendingRecording?: (pendingAt: number | null) => void
  // Called when the confirmation popup for stopping a recording is armed
  // ("save"/"discard") or resolved (null). Drives the in-headset popup.
  onPendingConfirm?: (action: ConfirmAction | null) => void
  // Called when the server announces its operating mode (once per connection).
  onMode?: (mode: AxolMode) => void
  // Called with the current 1-based episode number while collecting data (and
  // null if the server ever clears it). Drives the in-headset episode readout.
  onEpisode?: (episode: number | null) => void
  onExit?: () => void
}) {
  const { gl } = useThree()

  const stateRef = useRef<AxolState>(AxolState.Teleop)
  const seqRef = useRef(0)
  const poseSourceIdRef = useRef(
    `webxr-${globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)}`
  )
  const prevXRef = useRef(false)
  const prevYRef = useRef(false)
  const prevARef = useRef(false)
  const prevBRef = useRef(false)
  const recordingPendingAtRef = useRef<number | null>(null)
  // Which episode action (stop-to-save on A, discard on X) is awaiting a second
  // confirming press. Null when no confirmation popup is armed.
  const pendingConfirmRef = useRef<ConfirmAction | null>(null)
  // Server-pushed state override (e.g. "saving"). Applied at start of each frame.
  const serverStateRef = useRef<AxolState | null>(null)
  // Operating mode the server locked us to (null until announced). "teleop"
  // hides mode-toggle + recording; "data_collection" allows recording but not
  // toggling back to teleop.
  const modeRef = useRef<AxolMode | null>(null)
  // Server-pushed mode announcement, applied at the start of the next frame.
  const serverModeRef = useRef<AxolMode | null>(null)
  // Server-pushed episode number, applied at the start of the next frame. -1 is
  // the "unset" sentinel (distinct from a real episode value or an explicit
  // null the server could send); replaced with the parsed value on each push.
  const serverEpisodeRef = useRef<number | null | -1>(-1)
  // Track which WebSocket we have attached onmessage to avoid re-attaching.
  const wsWithHandlerRef = useRef<WebSocket | null>(null)
  // Change key of the last HUD state published to the server (see below).
  const lastHudKeyRef = useRef("")

  useFrame(() => {
    // Attach onmessage to the WebSocket whenever it changes so we can receive
    // server-pushed state overrides (e.g. the "saving" state after recording).
    const currentWs = wsRef.current
    if (currentWs !== wsWithHandlerRef.current) {
      wsWithHandlerRef.current = currentWs
      // A new (or dropped) connection invalidates the previous session's
      // episode number: the HUD readout only advances on a server `episode`
      // message, and plain teleop never sends one, so without this a prior
      // collect-data session's number would linger (even into a later teleop
      // session). Clear it here; a reconnecting collect-data session re-announces
      // the current episode on connect and repopulates it moments later.
      // Reset to the -1 "handled" sentinel (dropping any pending value) since we
      // notify here directly — the frame's apply block is gated behind an active
      // XR session, but a reconnect commonly happens outside one.
      serverEpisodeRef.current = -1
      onEpisode?.(null)
      if (currentWs) {
        currentWs.onmessage = (event: MessageEvent) => {
          try {
            const msg = JSON.parse(event.data as string) as {
              type: string
              value: string | number | null
            }
            if (msg.type === "state") {
              serverStateRef.current = msg.value as AxolState
            } else if (msg.type === "mode") {
              serverModeRef.current = msg.value as AxolMode
            } else if (msg.type === "episode") {
              serverEpisodeRef.current = typeof msg.value === "number" ? msg.value : null
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

    const state0 = stateRef.current

    function setState(next: AxolState) {
      stateRef.current = next
      onStateChange?.(next)
    }

    // Apply the server's mode announcement: lock the session to it. Teleop can
    // never enter data collection / recording; data collection can't fall back
    // to plain teleop. Done before the state override + button handling below.
    if (serverModeRef.current !== null) {
      const mode = serverModeRef.current
      serverModeRef.current = null
      modeRef.current = mode
      if (mode === "teleop") {
        if (state0 !== AxolState.Teleop) setState(AxolState.Teleop)
        recordingPendingAtRef.current = null
        onPendingRecording?.(null)
      } else if (mode === "data_collection" && state0 === AxolState.Teleop) {
        setState(AxolState.DataCollection)
      }
      onMode?.(mode)
    }

    // Surface any server-pushed episode number (purely informational — it
    // doesn't feed the state machine below).
    if (serverEpisodeRef.current !== -1) {
      onEpisode?.(serverEpisodeRef.current)
      serverEpisodeRef.current = -1
    }

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
    // Recording (and thus the data-collection state machine) only exists in
    // data-collection mode. In teleop mode A/B do nothing. Until the server
    // announces a mode we keep the legacy behaviour (recording allowed).
    const canRecord = modeRef.current !== "teleop"

    const isPending = recordingPendingAtRef.current !== null

    let reset = false

    if (state === AxolState.Recording && !isSaving) {
      // Stopping a recording — to save (A) or discard (X) — is gated by a
      // confirmation popup: the first press arms it, pressing the SAME button
      // again commits, the OTHER button cancels and keeps recording. Handled
      // before the generic A/X logic below so a stop can't slip through
      // unconfirmed.
      const pendingConfirm = pendingConfirmRef.current
      if (pendingConfirm === null) {
        if (aEdge) {
          pendingConfirmRef.current = "save"
          onPendingConfirm?.("save")
        } else if (xEdge) {
          pendingConfirmRef.current = "discard"
          onPendingConfirm?.("discard")
        }
      } else if (aEdge || xEdge) {
        const confirmed =
          (pendingConfirm === "save" && aEdge) || (pendingConfirm === "discard" && xEdge)
        if (confirmed) {
          // A discard carries the reset flag so the server drops the episode
          // and rewinds to re-record; a save stops with reset=false so it's
          // kept. Both leave the recording state.
          if (pendingConfirm === "discard") reset = true
          setState(AxolState.DataCollection)
        }
        // Same button commits, the other cancels — either way clear the popup.
        pendingConfirmRef.current = null
        onPendingConfirm?.(null)
      }
    } else {
      // Not recording: no episode to save/discard, so drop any stale armed
      // confirmation and fall back to the plain reset / start-recording logic.
      if (pendingConfirmRef.current !== null) {
        pendingConfirmRef.current = null
        onPendingConfirm?.(null)
      }

      // X — reset; also cancels a pending countdown (not allowed while saving).
      if (xEdge && !isSaving) {
        reset = true
        if (isPending) {
          setState(AxolState.DataCollection)
          recordingPendingAtRef.current = null
          onPendingRecording?.(null)
        }
      }

      // A — start pending (3s countdown) or cancel it; blocked while saving and
      // entirely unavailable in teleop mode (nothing records there).
      if (aEdge && !isSaving && canRecord) {
        if (state === AxolState.DataCollection && !isPending) {
          recordingPendingAtRef.current = Date.now()
          onPendingRecording?.(recordingPendingAtRef.current)
        } else if (isPending) {
          recordingPendingAtRef.current = null
          onPendingRecording?.(null)
        }
      }
    }

    // Y (left) — return home, disable, then exit XR. Piggy-back a reset on this
    // frame (same as X) so the backend plans a move back to rest and disengages
    // before we end the session; otherwise the arms hold the last commanded pose
    // and jerk when teleop is next entered. The session is ended only after the
    // frame carrying this reset is sent below (see the exit-on-send paths).
    if (yEdge) {
      reset = true
      // Abandon any armed confirmation on the way out of the session.
      if (pendingConfirmRef.current !== null) {
        pendingConfirmRef.current = null
        onPendingConfirm?.(null)
      }
    }

    // B (right) — handled in the app layer: it re-anchors the camera screens
    // to the current gaze (see App.tsx). Nothing is sent to the server for it.
    void bEdge

    // Promote pending → recording after 3s
    if (
      recordingPendingAtRef.current !== null &&
      Date.now() - recordingPendingAtRef.current >= 3000
    ) {
      setState(AxolState.Recording)
      recordingPendingAtRef.current = null
      onPendingRecording?.(null)
    }

    // Publish the HUD-only state (armed confirmation popup, record countdown)
    // to the server whenever it changes, so a dashboard watching the session
    // can mirror what this headset would show — the operator may be driving
    // with the controllers while the headset is off. Sent over the main
    // WebSocket (the signaling transport); the server relays it to the other
    // connected clients and clears it when we disconnect.
    {
      const pendingAt = recordingPendingAtRef.current
      const confirm = pendingConfirmRef.current
      const hudKey = `${confirm ?? ""}|${pendingAt !== null}`
      if (hudKey !== lastHudKeyRef.current) {
        lastHudKeyRef.current = hudKey
        const ws = wsRef.current
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "hud",
              value: {
                confirm,
                countdownRemainingMs:
                  pendingAt !== null ? Math.max(0, 3000 - (Date.now() - pendingAt)) : null,
              },
            })
          )
        }
      }
    }

    // Send each frame over every open transport: the wired USB `adb reverse`
    // tunnel for low latency, and the best network path (the WebRTC data
    // channel when open, else the main WebSocket) as an always-live standby.
    // The server de-dupes by seq and uses whichever arrives first — USB while
    // the cable is up, the network the instant USB goes quiet — so a USB drop
    // fails over with no reconnect. Bail early if neither is up so we skip
    // reading poses.
    const usbSink = pickUsbSink(poseWsRef)
    const netSink = pickNetworkSink(poseChannelRef, wsRef)
    if (usbSink == null && netSink == null) {
      // No transport to carry the reset — just leave XR (nothing to update).
      if (yEdge) onExit?.()
      return
    }

    function getPose(space: XRSpace | null | undefined) {
      if (!space) return null
      const pose = frame.getPose(space, refSpace!)
      if (!pose) return null
      const { position: p, orientation: o } = pose.transform
      return {
        pose: {
          position: { x: p.x, y: p.y, z: p.z },
          quaternion: { x: o.x, y: o.y, z: o.z, w: o.w },
        },
        // WebXR sets emulatedPosition while the controller's position is only
        // inertially dead-reckoned (occluded / out of camera view): it drifts
        // while coasting and snaps when optical tracking re-acquires. Ship the
        // flag so the server holds the last clean pose instead of chasing the
        // glitch. `!== true` keeps runtimes that omit the field as tracked.
        tracked: pose.emulatedPosition !== true,
      }
    }

    function getPosition(space: XRSpace | null | undefined) {
      if (!space) return null
      const pose = frame.getPose(space, refSpace!)
      if (!pose) return null
      const { position: p } = pose.transform
      return { x: p.x, y: p.y, z: p.z }
    }

    // A controller-mounted Mantis needs the physical grip datum, not WebXR's
    // runtime-defined aim ray (which vendors may move between firmware
    // releases). Older runtimes may omit gripSpace, so retain targetRaySpace
    // only as a compatibility fallback and surface that distinction during
    // calibration/setup.
    const l_pose_space = leftSource?.gripSpace ? "grip" : "target-ray"
    const r_pose_space = rightSource?.gripSpace ? "grip" : "target-ray"
    const l_hand = getPose(leftSource?.gripSpace ?? leftSource?.targetRaySpace)
    const r_hand = getPose(rightSource?.gripSpace ?? rightSource?.targetRaySpace)

    if (!l_hand || !r_hand) {
      // Lost controller tracking — can't ship a pose frame; leave XR anyway.
      if (yEdge) onExit?.()
      return
    }
    const l_ee = l_hand.pose
    const r_ee = r_hand.pose

    const body = (frame as XRFrame & { body?: XRBody }).body
    // Body tracking is optional on Quest and absolute Mantis mapping ignores
    // elbow hints. Keep controller tracking usable without it; relative Axol
    // teleop's default elbow weight is also zero, while operators who opt into
    // elbow weighting still get a stable (if neutral) controller-position
    // fallback instead of the entire pose stream silently stopping.
    const l_elbow = getPosition(body?.get(L_ELBOW_JOINT)) ?? l_ee.position
    const r_elbow = getPosition(body?.get(R_ELBOW_JOINT)) ?? r_ee.position

    const l_grip = 1 - (leftSource?.gamepad?.buttons[0]?.value ?? 0)
    const r_grip = 1 - (rightSource?.gamepad?.buttons[0]?.value ?? 0)
    const l_lock = (leftSource?.gamepad?.buttons[1]?.value ?? 0) >= 1.0
    const r_lock = (rightSource?.gamepad?.buttons[1]?.value ?? 0) >= 1.0

    // Thumbstick state for the powered cart (xr-standard mapping: stick axes
    // at axes[2]/[3], stick click at buttons[3]). Servers without a cart
    // configured simply ignore these fields.
    const l_stick_x = leftSource?.gamepad?.axes[2] ?? 0
    const l_stick_y = leftSource?.gamepad?.axes[3] ?? 0
    const r_stick_x = rightSource?.gamepad?.axes[2] ?? 0
    const l_stick_click = leftSource?.gamepad?.buttons[3]?.pressed ?? false
    const r_stick_click = rightSource?.gamepad?.buttons[3]?.pressed ?? false

    // Serialise once so both transports carry the identical frame (same seq),
    // letting the server treat them as one stream and de-dupe to whichever
    // arrives first.
    const payload = JSON.stringify({
      l_ee,
      r_ee,
      l_elbow,
      r_elbow,
      l_lock,
      r_lock,
      l_grip,
      r_grip,
      // Per-hand optical-tracking state (see getPose): the server holds a
      // hand's last tracked pose while its flag is false.
      l_tracked: l_hand.tracked,
      r_tracked: r_hand.tracked,
      reset,
      state: stateRef.current,
      l_stick_x,
      l_stick_y,
      r_stick_x,
      l_stick_click,
      r_stick_click,
      seq: ++seqRef.current,
      // One logical identity across USB, WebRTC, and network WebSocket. The
      // server de-dupes the shared sequence globally and can keep this Quest
      // view-only when a Lighthouse/Ultimate bridge owns Mantis poses.
      pose_source_id: poseSourceIdRef.current,
      pose_source_kind: "webxr",
      // Mount calibration is datum-specific. The host validates both the
      // WebXR controller generation and grip-vs-aim space before accepting a
      // production Quest pose.
      l_pose_profile: leftSource?.profiles?.[0] ?? null,
      r_pose_profile: rightSource?.profiles?.[0] ?? null,
      l_pose_space,
      r_pose_space,
      // Capture timestamp (ms). The server's pose interpolator uses this to
      // reconstruct the true motion cadence when frames arrive batched over a
      // jittery/relayed link, so teleop stays smooth instead of stuttering.
      t: performance.now(),
    })
    usbSink?.send(payload)
    netSink?.send(payload)

    // End the XR session only now that the Y-press reset frame has been sent, so
    // the backend receives the return-to-rest before the pose stream stops.
    if (yEdge) {
      // This exit frame carries the return-to-rest reset and is the last one
      // sent — no later frame can re-carry it. The preferred pose channel is
      // deliberately unreliable (maxRetransmits: 0), so one dropped datagram
      // would leave VR with the robot still engaged mid-air; duplicate the
      // frame over the reliable WebSocket too (the server de-dupes by seq).
      const ws = wsRef.current
      if (ws && ws.readyState === WebSocket.OPEN && netSink !== ws) {
        ws.send(payload)
      }
      onExit?.()
    }
  })

  return null
}
