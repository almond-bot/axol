import type { RefObject } from "react"
import { useEffect, useRef, useState } from "react"
import { AxolConnectionStatus } from "./types"
import { waitForIceGathering } from "./webrtc"

const POLL_MS = 300
// After a failed negotiation, wait this long before re-requesting the channel
// (the WebSocket pose path covers us meanwhile, so don't hammer it).
const RETRY_MS = 3000

/**
 * Negotiates a low-latency WebRTC data channel for pose frames, multiplexing
 * signaling over the existing teleop WebSocket (same pattern as `useAxolVideo`).
 *
 * Pose frames default to the WebSocket, but over a Tailscale Funnel that path is
 * relayed TCP — head-of-line blocking makes the high-rate pose stream lag. This
 * channel is **unreliable + unordered** (configured server-side) and rides the
 * same UDP / ICE / TURN path as the camera video, so control latency drops to
 * the video's level. On a LAN it negotiates a direct host candidate. The
 * WebSocket stays as the fallback whenever the channel isn't open.
 *
 * Returns a ref to the open `RTCDataChannel` (or null) for `AxolVRClient` to
 * prefer, plus a status. Enable it whenever the teleop connection is up — it's
 * independent of whether cameras are streaming.
 */
export function useAxolControlChannel(
  wsRef: RefObject<WebSocket | null>,
  enabled: boolean
): { poseChannelRef: RefObject<RTCDataChannel | null>; status: AxolConnectionStatus } {
  const [status, setStatus] = useState<AxolConnectionStatus>(AxolConnectionStatus.Idle)

  const pcRef = useRef<RTCPeerConnection | null>(null)
  const poseChannelRef = useRef<RTCDataChannel | null>(null)
  const attachedWsRef = useRef<WebSocket | null>(null)
  const listenerRef = useRef<((e: MessageEvent) => void) | null>(null)
  const requestedRef = useRef(false)
  // Timestamp (ms) at which a failed negotiation may be retried, or null.
  const retryAtRef = useRef<number | null>(null)

  useEffect(() => {
    if (!enabled) return

    function closePc() {
      if (poseChannelRef.current) {
        try {
          poseChannelRef.current.close()
        } catch {
          // already closed
        }
        poseChannelRef.current = null
      }
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
      retryAtRef.current = null
      closePc()
      setStatus(AxolConnectionStatus.Idle)
    }

    async function handleOffer(sdp: string, iceServers: RTCIceServer[]) {
      closePc()
      // Off-LAN operators get TURN/STUN from the server; on a LAN the list is
      // empty and the browser default (direct host candidate) is used.
      const pc = new RTCPeerConnection(iceServers.length > 0 ? { iceServers } : undefined)
      pcRef.current = pc
      setStatus(AxolConnectionStatus.Connecting)

      pc.ondatachannel = (e: RTCDataChannelEvent) => {
        if (e.channel.label !== "pose") return
        const ch = e.channel
        poseChannelRef.current = ch
        ch.onopen = () => setStatus(AxolConnectionStatus.Open)
        ch.onclose = () => {
          if (poseChannelRef.current === ch) poseChannelRef.current = null
        }
      }
      pc.onconnectionstatechange = () => {
        // Ignore events from a superseded pc (e.g. closePc during a retry),
        // so intentionally tearing one down doesn't schedule a spurious retry.
        if (pcRef.current !== pc) return
        if (pc.connectionState === "failed" || pc.connectionState === "closed") {
          poseChannelRef.current = null
          setStatus(AxolConnectionStatus.Error)
          // Let the poll loop re-request a fresh channel after a backoff; a
          // transient ICE/SDP failure shouldn't permanently block the channel.
          retryAtRef.current = Date.now() + RETRY_MS
        }
      }

      await pc.setRemoteDescription({ type: "offer", sdp })
      const answer = await pc.createAnswer()
      await pc.setLocalDescription(answer)
      // Non-trickle: wait for gathering so the (TURN relay) candidate is in the
      // answer SDP. On a LAN this completes immediately.
      await waitForIceGathering(pc)
      const answerSdp = pc.localDescription?.sdp ?? answer.sdp
      attachedWsRef.current?.send(JSON.stringify({ type: "control-answer", sdp: answerSdp }))
    }

    function onMessage(e: MessageEvent) {
      let msg: unknown
      try {
        msg = JSON.parse(e.data as string)
      } catch {
        return
      }
      if (typeof msg !== "object" || msg === null) return
      const m = msg as { type?: string; sdp?: string; iceServers?: RTCIceServer[] }
      if (m.type === "control-offer" && typeof m.sdp === "string") {
        handleOffer(m.sdp, m.iceServers ?? []).catch(() => {
          setStatus(AxolConnectionStatus.Error)
          retryAtRef.current = Date.now() + RETRY_MS
        })
      }
    }

    // The WebSocket is owned elsewhere and may be (re)created at any time, so we
    // poll for it: attach our listener and request the channel once it's open,
    // and re-arm whenever the socket instance changes.
    const interval = setInterval(() => {
      const ws = wsRef.current
      if (ws !== attachedWsRef.current) {
        if (attachedWsRef.current && listenerRef.current)
          attachedWsRef.current.removeEventListener("message", listenerRef.current)
        attachedWsRef.current = null
        listenerRef.current = null
        requestedRef.current = false
        retryAtRef.current = null
        closePc()
        setStatus(AxolConnectionStatus.Idle)
      }
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      if (!listenerRef.current) {
        listenerRef.current = onMessage
        attachedWsRef.current = ws
        ws.addEventListener("message", onMessage)
      }
      // A prior negotiation failed; once the backoff elapses, allow a re-request.
      if (retryAtRef.current !== null && Date.now() >= retryAtRef.current) {
        retryAtRef.current = null
        requestedRef.current = false
      }
      if (!requestedRef.current) {
        requestedRef.current = true
        try {
          ws.send(JSON.stringify({ type: "control-request" }))
        } catch {
          requestedRef.current = false
        }
      }
    }, POLL_MS)

    return () => {
      clearInterval(interval)
      detach()
    }
  }, [enabled, wsRef])

  return { poseChannelRef, status }
}
