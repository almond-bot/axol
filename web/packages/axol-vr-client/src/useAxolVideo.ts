import type { RefObject } from "react"
import { useEffect, useRef, useState } from "react"

/** Live camera streams keyed by camera name (e.g. "overhead", "left_arm"). */
export type CameraStreams = Record<string, MediaStream>

const POLL_MS = 300

// Cap on how long we wait for ICE gathering before sending the answer. Host
// candidates gather almost instantly; a TURN allocation is a quick round-trip.
// The cap just prevents a stalled TURN server from hanging negotiation forever
// (we then send whatever candidates we have).
const ICE_GATHER_TIMEOUT_MS = 3000

/**
 * Resolve once the peer connection has finished gathering ICE candidates, so a
 * non-trickle answer carries them all (crucially the TURN relay candidate).
 * Falls back after `ICE_GATHER_TIMEOUT_MS` if gathering stalls.
 */
function waitForIceGathering(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve()
  return new Promise((resolve) => {
    const finish = () => {
      pc.removeEventListener("icegatheringstatechange", onChange)
      clearTimeout(timer)
      resolve()
    }
    const onChange = () => {
      if (pc.iceGatheringState === "complete") finish()
    }
    const timer = setTimeout(finish, ICE_GATHER_TIMEOUT_MS)
    pc.addEventListener("icegatheringstatechange", onChange)
  })
}

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
 * `enabled` gates negotiation — pass `true` only while the headset is presenting
 * so video isn't decoded on the 2D landing page. Returns the current streams and
 * `available`: `null` until known, `false` if the server reports no video.
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
  const requestedRef = useRef(false)

  useEffect(() => {
    if (!enabled) return

    function closePc() {
      if (pcRef.current) {
        try {
          pcRef.current.close()
        } catch {
          // already closed
        }
        pcRef.current = null
      }
    }

    function detach() {
      const ws = attachedWsRef.current
      if (ws && listenerRef.current) ws.removeEventListener("message", listenerRef.current)
      attachedWsRef.current = null
      listenerRef.current = null
      requestedRef.current = false
      closePc()
    }

    async function handleOffer(
      sdp: string,
      trackMap: Record<string, string>,
      iceServers: RTCIceServer[]
    ) {
      closePc()
      // Off-LAN operators get TURN/STUN servers from the server; on a LAN the
      // list is empty and we use the browser default (direct host candidates).
      const pc = new RTCPeerConnection(iceServers.length > 0 ? { iceServers } : undefined)
      pcRef.current = pc

      // Accumulate streams as tracks arrive, matching each transceiver's mid to
      // its camera name from the server's map.
      const acc: CameraStreams = {}
      pc.ontrack = (e: RTCTrackEvent) => {
        // Ask the receiver to render with minimal buffering: this is a live
        // LAN teleop feed, so trade jitter resilience for latency (Chromium
        // jitter buffers can otherwise hold frames for tens of ms).
        const receiver = e.receiver as RTCRtpReceiver & {
          playoutDelayHint?: number
          jitterBufferTarget?: number | null
        }
        try {
          receiver.jitterBufferTarget = 0 // standard, milliseconds
          receiver.playoutDelayHint = 0 // legacy Chromium, seconds
        } catch {
          // best-effort; older browsers may reject the setters
        }
        const mid = e.transceiver?.mid ?? null
        const name = mid != null ? trackMap[mid] : undefined
        if (!name) return
        acc[name] = new MediaStream([e.track])
        setStreams({ ...acc })
      }
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "failed" || pc.connectionState === "closed") {
          setStreams({})
        }
      }

      await pc.setRemoteDescription({ type: "offer", sdp })
      const answer = await pc.createAnswer()
      await pc.setLocalDescription(answer)
      // Non-trickle: wait for gathering so the (TURN relay) candidate is in the
      // SDP we signal back. On a LAN this completes immediately.
      await waitForIceGathering(pc)
      const answerSdp = pc.localDescription?.sdp ?? answer.sdp
      attachedWsRef.current?.send(JSON.stringify({ type: "webrtc-answer", sdp: answerSdp }))
    }

    function onMessage(e: MessageEvent) {
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
        setAvailable(true)
        handleOffer(m.sdp, m.tracks ?? {}, m.iceServers ?? []).catch(() => {
          /* negotiation failed; leave streams empty */
        })
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
        requestedRef.current = false
        closePc()
        setStreams({})
        setAvailable(null)
      }
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      if (!listenerRef.current) {
        listenerRef.current = onMessage
        attachedWsRef.current = ws
        ws.addEventListener("message", onMessage)
      }
      if (!requestedRef.current) {
        requestedRef.current = true
        try {
          ws.send(JSON.stringify({ type: "webrtc-request" }))
        } catch {
          requestedRef.current = false
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
