import type { RefObject } from "react"
import { useEffect, useRef, useState } from "react"
import {
  controlPeerNeedsRetry,
  controlRequestIsDue,
  retireCurrentControlPeer,
} from "./controlRetry"
import { AxolConnectionStatus } from "./types"
import { waitForIceGathering } from "./webrtc"

const POLL_MS = 300
// Keep requesting until an offer arrives. A server-side createOffer task can
// fail before it has anything to send, and an older server has no explicit
// control-error reply. The WebSocket pose path covers this modest retry cadence.
const REQUEST_RETRY_MS = 3000

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
  const lastRequestAtRef = useRef(0)
  const offerSeenRef = useRef(false)

  useEffect(() => {
    if (!enabled) return

    function closePc() {
      const channel = poseChannelRef.current
      const pc = pcRef.current
      // Retire first: close() may synchronously dispatch a terminal callback,
      // and intentional teardown must not schedule another negotiation.
      poseChannelRef.current = null
      pcRef.current = null
      if (channel) {
        try {
          channel.close()
        } catch {
          // already closed
        }
      }
      if (pc) {
        try {
          pc.close()
        } catch {
          // already closed
        }
      }
    }

    function rearmPeer(signalingWs: WebSocket, pc: RTCPeerConnection) {
      if (!retireCurrentControlPeer(attachedWsRef, pcRef, signalingWs, pc)) return
      poseChannelRef.current = null
      try {
        pc.close()
      } catch {
        // already closed
      }
      setStatus(AxolConnectionStatus.Error)
      offerSeenRef.current = false
      lastRequestAtRef.current = 0
    }

    function rearmSocket(signalingWs: WebSocket) {
      if (attachedWsRef.current !== signalingWs) return
      closePc()
      setStatus(AxolConnectionStatus.Error)
      offerSeenRef.current = false
      lastRequestAtRef.current = 0
    }

    function detach() {
      const ws = attachedWsRef.current
      if (ws && listenerRef.current) ws.removeEventListener("message", listenerRef.current)
      attachedWsRef.current = null
      listenerRef.current = null
      lastRequestAtRef.current = 0
      offerSeenRef.current = false
      closePc()
      setStatus(AxolConnectionStatus.Idle)
    }

    async function handleOffer(signalingWs: WebSocket, sdp: string, iceServers: RTCIceServer[]) {
      // An offer event can already be queued when the signaling socket is
      // replaced. Never let that old negotiation tear down or answer through
      // the new socket.
      if (attachedWsRef.current !== signalingWs) return
      closePc()
      // Off-LAN operators get TURN/STUN from the server; on a LAN the list is
      // empty and the browser default (direct host candidate) is used.
      const pc = new RTCPeerConnection(iceServers.length > 0 ? { iceServers } : undefined)
      pcRef.current = pc
      setStatus(AxolConnectionStatus.Connecting)
      const isCurrent = () => attachedWsRef.current === signalingWs && pcRef.current === pc

      pc.ondatachannel = (e: RTCDataChannelEvent) => {
        if (!isCurrent()) return
        if (e.channel.label !== "pose") return
        const ch = e.channel
        poseChannelRef.current = ch
        ch.onopen = () => {
          if (isCurrent() && poseChannelRef.current === ch) {
            setStatus(AxolConnectionStatus.Open)
          }
        }
        ch.onclose = () => {
          // A data channel can close while its peer remains "connected". It
          // cannot be recreated without a new server offer, so retire this peer.
          if (isCurrent() && poseChannelRef.current === ch) rearmPeer(signalingWs, pc)
        }
      }
      pc.onconnectionstatechange = () => {
        // Ignore events from a superseded pc (e.g. closePc during a retry),
        // so intentionally tearing one down doesn't schedule a spurious retry.
        if (!isCurrent()) return
        if (controlPeerNeedsRetry(pc.connectionState)) rearmPeer(signalingWs, pc)
      }

      try {
        await pc.setRemoteDescription({ type: "offer", sdp })
        if (!isCurrent()) return
        const answer = await pc.createAnswer()
        if (!isCurrent()) return
        await pc.setLocalDescription(answer)
        if (!isCurrent()) return
        // Non-trickle: wait for gathering so the (TURN relay) candidate is in
        // the answer SDP. On a LAN this completes immediately.
        await waitForIceGathering(pc)
        if (!isCurrent()) return
        const answerSdp = pc.localDescription?.sdp ?? answer.sdp
        signalingWs.send(JSON.stringify({ type: "control-answer", sdp: answerSdp }))
      } catch {
        if (!isCurrent()) return
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
      const m = msg as { type?: string; sdp?: string; iceServers?: RTCIceServer[] }
      if (m.type === "control-offer" && typeof m.sdp === "string") {
        offerSeenRef.current = true
        void handleOffer(signalingWs, m.sdp, m.iceServers ?? [])
      } else if (m.type === "control-error") {
        // Current servers report an offer task that failed before producing SDP.
        // Older servers stay covered by the periodic no-offer retry below.
        rearmSocket(signalingWs)
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
        lastRequestAtRef.current = 0
        offerSeenRef.current = false
        closePc()
        setStatus(AxolConnectionStatus.Idle)
      }
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      if (!listenerRef.current) {
        const listener = (event: MessageEvent) => onMessage(ws, event)
        listenerRef.current = listener
        attachedWsRef.current = ws
        ws.addEventListener("message", listener)
      }
      const now = Date.now()
      if (
        controlRequestIsDue(offerSeenRef.current, lastRequestAtRef.current, now, REQUEST_RETRY_MS)
      ) {
        lastRequestAtRef.current = now
        try {
          ws.send(JSON.stringify({ type: "control-request" }))
        } catch {
          lastRequestAtRef.current = 0
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
