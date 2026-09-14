import type { RefObject } from "react"
import { useEffect, useRef, useState } from "react"
import {
  retireCurrentVideoPeer,
  videoDisconnectRearmDelay,
  videoPeerNeedsRetry,
  videoRequestIsDue,
} from "./videoRetry"
import { waitForIceGathering } from "./webrtc"

/** Live camera streams keyed by camera name (e.g. "overhead", "left_arm"). */
export type CameraStreams = Record<string, MediaStream>

const POLL_MS = 300

// Re-send `webrtc-request` at this cadence until an offer arrives. The
// robot's cameras can take a while to start after its server begins accepting
// connections; during that window the server answers "webrtc-pending" (new
// servers, which also push the offer when ready) or "webrtc-unavailable" (old
// servers, which never retry on their own) — either way, without a retry the
// feed would stay absent for the whole session even once the cameras are up.
// The request is a tiny JSON message on the already-open teleop socket, so
// retrying is harmless even against a robot that genuinely has no cameras.
const REQUEST_RETRY_MS = 2000

// After a peer is re-armed because negotiation *failed* (rejected SDP, a peer
// that went failed/closed, a track that ended), hold the next request back by
// this much. A socket swap still requests immediately; only failures back off,
// so a fast deterministic failure can't become a 300 ms loop of server
// create_offer/close cycles.
const FAILURE_BACKOFF_MS = REQUEST_RETRY_MS

// A peer in "disconnected" gets this long to come back on its own before it is
// torn down and renegotiated. Per the WebRTC spec the state is transient: over
// Tailscale Funnel/TURN or a Wi-Fi roam Chrome flaps connected → disconnected
// → connected in ~1–3 s, and a full renegotiation (server close, new offer,
// ICE/TURN, DTLS, first keyframe) costs far more frozen video than waiting.
const DISCONNECT_GRACE_MS = 5000

// Receiver jitter-buffer target (ms). A *zero* buffer minimises latency but
// leaves no time for a NACK retransmit of a dropped/reordered RTP packet, so
// the decoder presents an incomplete frame — corrupt macroblocks that flash as
// a garbled band (usually near the top of the image) until the next keyframe
// ~1s later. A small buffer lets the retransmit land before playout and clears
// that up. Kept low for LAN teleop, where the NACK round trip is ~1ms so this
// recovers essentially all loss; raise it on a lossier link, lower toward 0 to
// shave latency at the cost of the tearing returning.
const JITTER_BUFFER_MS = 100

/**
 * Negotiates a WebRTC connection that receives the Axol cameras and exposes
 * them as `MediaStream`s keyed by camera name.
 *
 * Signaling is multiplexed over the existing teleop WebSocket (no new ports):
 * we send `{type:"webrtc-request"}`, the server answers with an SDP offer plus a
 * `mid → cameraName` map (and, for off-LAN operators, the `iceServers` to use),
 * and we reply with an SDP answer. ICE is non-trickle (candidates are embedded
 * in the SDP), so on a LAN no candidate exchange is needed; with a TURN server
 * we must wait for gathering to finish so our relay candidate lands in the
 * answer SDP.
 *
 * A `message` *listener* is used (not `ws.onmessage`) so this coexists with the
 * pose client's own `onmessage` handler on the same socket.
 *
 * The request is retried until an offer lands (see REQUEST_RETRY_MS), and a
 * failed/closed peer connection clears the negotiated state so the retry loop
 * renegotiates — the feed recovers on its own once the robot's cameras come
 * (back) up instead of requiring the operator to leave and re-enter VR.
 *
 * `enabled` gates negotiation — pass `true` only while the headset is presenting
 * so video isn't decoded on the 2D landing page. Returns the current streams and
 * `available`: `null` until known (or while the server reports video is still
 * starting via `webrtc-pending`), `false` if the server reports no video.
 */
export function useAxolVideo(
  wsRef: RefObject<WebSocket | null>,
  enabled: boolean
): { streams: CameraStreams; available: boolean | null } {
  const [streams, setStreams] = useState<CameraStreams>({})
  const [available, setAvailable] = useState<boolean | null>(null)

  const pcRef = useRef<RTCPeerConnection | null>(null)
  const attachedWsRef = useRef<WebSocket | null>(null)
  const listenerRef = useRef<((e: MessageEvent) => void) | null>(null)
  // When the last webrtc-request was sent (0 = not yet), and whether an offer
  // has landed for it. Requests keep re-sending until an offer arrives, and a
  // dead peer connection clears `offerSeen` so the loop renegotiates.
  const lastRequestAtRef = useRef(0)
  const offerSeenRef = useRef(false)
  // Earliest time the next webrtc-request may go out; pushed forward by
  // FAILURE_BACKOFF_MS when a rearm is caused by a failure.
  const requestNotBeforeRef = useRef(0)
  // Pending grace timer for the current peer's "disconnected" state.
  const disconnectGraceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!enabled) return

    function clearDisconnectGrace() {
      if (disconnectGraceRef.current !== null) {
        clearTimeout(disconnectGraceRef.current)
        disconnectGraceRef.current = null
      }
    }

    function closePc() {
      clearDisconnectGrace()
      const pc = pcRef.current
      // Retire before close(): Chromium may dispatch "closed" synchronously,
      // and that local teardown must not re-arm negotiation.
      pcRef.current = null
      if (pc) {
        try {
          pc.close()
        } catch {
          // already closed
        }
      }
    }

    // Retire a peer whose negotiation *failed* (remote failure, rejected SDP,
    // ended track, or a disconnect that outlived its grace) and let the polling
    // loop request a fresh offer — after FAILURE_BACKOFF_MS, not on the next
    // tick, so a deterministic failure doesn't hammer the server.
    function rearmPeer(signalingWs: WebSocket, pc: RTCPeerConnection) {
      if (!retireCurrentVideoPeer(attachedWsRef, pcRef, signalingWs, pc)) return
      clearDisconnectGrace()
      try {
        pc.close()
      } catch {
        // already closed
      }
      setStreams({})
      setAvailable(null)
      offerSeenRef.current = false
      lastRequestAtRef.current = 0
      requestNotBeforeRef.current = Date.now() + FAILURE_BACKOFF_MS
    }

    function detach() {
      const ws = attachedWsRef.current
      if (ws && listenerRef.current) ws.removeEventListener("message", listenerRef.current)
      attachedWsRef.current = null
      listenerRef.current = null
      lastRequestAtRef.current = 0
      requestNotBeforeRef.current = 0
      offerSeenRef.current = false
      closePc()
    }

    async function handleOffer(
      signalingWs: WebSocket,
      sdp: string,
      trackMap: Record<string, string>,
      iceServers: RTCIceServer[]
    ) {
      // A queued offer from a socket that was just replaced must not close the
      // new socket's peer connection or send its SDP answer to the new host.
      if (attachedWsRef.current !== signalingWs) return
      closePc()
      // Off-LAN operators get TURN/STUN servers from the server; on a LAN the
      // list is empty and we use the browser default (direct host candidates).
      const pc = new RTCPeerConnection(iceServers.length > 0 ? { iceServers } : undefined)
      pcRef.current = pc
      const isCurrent = () => attachedWsRef.current === signalingWs && pcRef.current === pc

      // Accumulate streams as tracks arrive, matching each transceiver's mid to
      // its camera name from the server's map.
      const acc: CameraStreams = {}
      pc.ontrack = (e: RTCTrackEvent) => {
        if (!isCurrent()) return
        // Keep the receiver buffer small for low-latency LAN teleop, but not
        // zero: a tiny buffer still lets a NACK retransmit recover a lost or
        // reordered packet before playout, instead of decoding an incomplete
        // frame and flashing corrupt macroblocks until the next keyframe.
        const receiver = e.receiver as RTCRtpReceiver & {
          playoutDelayHint?: number
          jitterBufferTarget?: number | null
        }
        try {
          receiver.jitterBufferTarget = JITTER_BUFFER_MS // standard, milliseconds
          receiver.playoutDelayHint = JITTER_BUFFER_MS / 1000 // legacy Chromium, seconds
        } catch {
          // best-effort; older browsers may reject the setters
        }
        const mid = e.transceiver?.mid ?? null
        const name = mid != null ? trackMap[mid] : undefined
        if (!name) return
        e.track.addEventListener("ended", () => rearmPeer(signalingWs, pc), {
          once: true,
        })
        acc[name] = new MediaStream([e.track])
        setStreams({ ...acc })
      }
      // "disconnected" is transient: give it DISCONNECT_GRACE_MS to recover
      // before renegotiating. Any state change cancels the pending grace timer;
      // if the state is still "disconnected" when it fires, rearm.
      const scheduleDisconnectRearm = (disconnectedAt: number, delayMs: number) => {
        disconnectGraceRef.current = setTimeout(() => {
          disconnectGraceRef.current = null
          if (!isCurrent()) return
          const remaining = videoDisconnectRearmDelay(
            pc.connectionState,
            disconnectedAt,
            Date.now(),
            DISCONNECT_GRACE_MS
          )
          if (remaining === null) return
          if (remaining > 0) {
            scheduleDisconnectRearm(disconnectedAt, remaining)
            return
          }
          rearmPeer(signalingWs, pc)
        }, delayMs)
      }
      pc.onconnectionstatechange = () => {
        if (!isCurrent()) return
        clearDisconnectGrace()
        const state = pc.connectionState
        if (videoPeerNeedsRetry(state)) {
          rearmPeer(signalingWs, pc)
          return
        }
        const now = Date.now()
        const delay = videoDisconnectRearmDelay(state, now, now, DISCONNECT_GRACE_MS)
        if (delay !== null) scheduleDisconnectRearm(now, delay)
      }

      try {
        await pc.setRemoteDescription({ type: "offer", sdp })
        if (!isCurrent()) return
        const answer = await pc.createAnswer()
        if (!isCurrent()) return
        await pc.setLocalDescription(answer)
        if (!isCurrent()) return
        // Non-trickle: wait for gathering so the (TURN relay) candidate is in
        // the SDP we signal back. On a LAN this completes immediately.
        await waitForIceGathering(pc)
        if (!isCurrent()) return
        const answerSdp = pc.localDescription?.sdp ?? answer.sdp
        signalingWs.send(JSON.stringify({ type: "webrtc-answer", sdp: answerSdp }))
      } catch {
        if (!isCurrent()) return
        // Negotiation failed; let the retry loop request a fresh offer.
        rearmPeer(signalingWs, pc)
      }
    }

    function onMessage(signalingWs: WebSocket, e: MessageEvent) {
      if (attachedWsRef.current !== signalingWs) return
      let msg: unknown
      try {
        msg = JSON.parse(e.data as string)
      } catch {
        return
      }
      if (typeof msg !== "object" || msg === null) return
      const m = msg as {
        type?: string
        sdp?: string
        tracks?: Record<string, string>
        iceServers?: RTCIceServer[]
      }
      if (m.type === "webrtc-offer" && typeof m.sdp === "string") {
        offerSeenRef.current = true
        setAvailable(true)
        void handleOffer(signalingWs, m.sdp, m.tracks ?? {}, m.iceServers ?? [])
      } else if (m.type === "webrtc-pending") {
        // The robot's cameras are configured but still starting; the server
        // will push the offer when they're up. Stay in the "connecting"
        // state (spinner visible) rather than reporting no video.
        setAvailable(null)
      } else if (m.type === "webrtc-unavailable") {
        setAvailable(false)
      }
    }

    // The WebSocket is owned elsewhere and may be (re)created at any time, so we
    // poll for it: attach our listener and kick off signaling once it's open,
    // and re-arm whenever the socket instance changes.
    const interval = setInterval(() => {
      const ws = wsRef.current
      if (ws !== attachedWsRef.current) {
        if (attachedWsRef.current && listenerRef.current)
          attachedWsRef.current.removeEventListener("message", listenerRef.current)
        attachedWsRef.current = null
        listenerRef.current = null
        // A fresh socket requests right away: no failure backoff applies.
        lastRequestAtRef.current = 0
        requestNotBeforeRef.current = 0
        offerSeenRef.current = false
        closePc()
        setStreams({})
        setAvailable(null)
      }
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      if (!listenerRef.current) {
        const listener = (event: MessageEvent) => onMessage(ws, event)
        listenerRef.current = listener
        attachedWsRef.current = ws
        ws.addEventListener("message", listener)
      }
      // Keep requesting until an offer lands (first request fires
      // immediately; see REQUEST_RETRY_MS for why this retries, and
      // FAILURE_BACKOFF_MS for why a failed negotiation waits).
      const now = Date.now()
      if (
        videoRequestIsDue(
          offerSeenRef.current,
          lastRequestAtRef.current,
          now,
          REQUEST_RETRY_MS,
          requestNotBeforeRef.current
        )
      ) {
        lastRequestAtRef.current = now
        try {
          ws.send(JSON.stringify({ type: "webrtc-request" }))
        } catch {
          lastRequestAtRef.current = 0
        }
      }
    }, POLL_MS)

    return () => {
      clearInterval(interval)
      detach()
      setStreams({})
      setAvailable(null)
    }
  }, [enabled, wsRef])

  return { streams, available }
}
