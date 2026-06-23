import type { RefObject } from "react"
import { useEffect, useRef } from "react"

const POLL_MS = 300
const ICE_GATHERING_TIMEOUT_MS = 2000
const STATS_MS = 200

type StatsRecord = RTCStats & Record<string, unknown>

function statNumber(stat: StatsRecord | undefined, key: string): number | undefined {
  const value = stat?.[key]
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function statString(stat: StatsRecord | undefined, key: string): string | undefined {
  const value = stat?.[key]
  return typeof value === "string" ? value : undefined
}

function candidateLabel(stat: StatsRecord | undefined): string | undefined {
  if (!stat) return undefined
  const address = statString(stat, "address") ?? statString(stat, "ip") ?? "?"
  const port = statNumber(stat, "port")
  const protocol = statString(stat, "protocol") ?? "?"
  const candidateType = statString(stat, "candidateType") ?? "?"
  const networkType = statString(stat, "networkType")
  return `${address}${port ? `:${port}` : ""}/${protocol}/${candidateType}${networkType ? `/${networkType}` : ""}`
}

function waitForIceGatheringComplete(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve()
  return new Promise((resolve) => {
    const timeout = setTimeout(done, ICE_GATHERING_TIMEOUT_MS)
    function done() {
      clearTimeout(timeout)
      pc.removeEventListener("icegatheringstatechange", onChange)
      resolve()
    }
    function onChange() {
      if (pc.iceGatheringState === "complete") done()
    }
    pc.addEventListener("icegatheringstatechange", onChange)
  })
}

/**
 * Negotiates an unreliable unordered WebRTC data channel for high-rate pose
 * frames. The existing WebSocket remains only for SDP signaling and low-rate
 * server feedback; once this ref is open, pose transport is UDP/SCTP rather
 * than reliable ordered WebSocket/TCP.
 */
export function useAxolPoseChannel(
  wsRef: RefObject<WebSocket | null>
): RefObject<RTCDataChannel | null> {
  const channelRef = useRef<RTCDataChannel | null>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const attachedWsRef = useRef<WebSocket | null>(null)
  const listenerRef = useRef<((e: MessageEvent) => void) | null>(null)
  const requestedRef = useRef(false)
  const connectingRef = useRef(false)

  useEffect(() => {
    function closePc() {
      const channel = channelRef.current
      if (channel) {
        channel.onopen = null
        channel.onclose = null
        channel.onerror = null
        try {
          channel.close()
        } catch {
          // already closed
        }
        channelRef.current = null
      }
      if (pcRef.current) {
        pcRef.current.onconnectionstatechange = null
        try {
          pcRef.current.close()
        } catch {
          // already closed
        }
        pcRef.current = null
      }
      connectingRef.current = false
    }

    function detach() {
      const ws = attachedWsRef.current
      if (ws && listenerRef.current) ws.removeEventListener("message", listenerRef.current)
      attachedWsRef.current = null
      listenerRef.current = null
      requestedRef.current = false
      closePc()
    }

    async function start(ws: WebSocket) {
      if (connectingRef.current) return
      closePc()
      connectingRef.current = true

      const pc = new RTCPeerConnection()
      pcRef.current = pc
      const channel = pc.createDataChannel("pose", {
        ordered: false,
        maxPacketLifeTime: 20,
      })
      channelRef.current = channel

      channel.onclose = () => {
        if (channelRef.current === channel) channelRef.current = null
      }
      channel.onerror = () => {
        if (channelRef.current === channel) channelRef.current = null
      }
      pc.onconnectionstatechange = () => {
        if (
          pc.connectionState === "failed" ||
          pc.connectionState === "closed" ||
          pc.connectionState === "disconnected"
        ) {
          closePc()
          requestedRef.current = false
        }
      }

      try {
        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await waitForIceGatheringComplete(pc)
        if (ws.readyState !== WebSocket.OPEN) {
          requestedRef.current = false
          closePc()
          return
        }
        ws.send(
          JSON.stringify({
            type: "pose-webrtc-offer",
            sdp: pc.localDescription?.sdp ?? offer.sdp,
          })
        )
      } catch {
        requestedRef.current = false
        closePc()
      } finally {
        connectingRef.current = false
      }
    }

    async function sendStats(ws: WebSocket) {
      const pc = pcRef.current
      const channel = channelRef.current
      if (!pc || !channel || ws.readyState !== WebSocket.OPEN) return

      try {
        const report = await pc.getStats()
        let selectedPair: StatsRecord | undefined
        let fallbackPair: StatsRecord | undefined
        let dataChannelStats: StatsRecord | undefined

        report.forEach((raw) => {
          const stat = raw as StatsRecord
          if (stat.type === "candidate-pair") {
            const state = statString(stat, "state")
            const nominated = stat["nominated"] === true
            const selected = stat["selected"] === true
            if (selected) selectedPair = stat
            if (!fallbackPair && nominated && state === "succeeded") fallbackPair = stat
          } else if (stat.type === "data-channel" && statString(stat, "label") === "pose") {
            dataChannelStats = stat
          }
        })

        const pair = selectedPair ?? fallbackPair
        const local = pair
          ? (report.get(statString(pair, "localCandidateId") ?? "") as StatsRecord | undefined)
          : undefined
        const remote = pair
          ? (report.get(statString(pair, "remoteCandidateId") ?? "") as StatsRecord | undefined)
          : undefined
        const rtt = statNumber(pair, "currentRoundTripTime")

        ws.send(
          JSON.stringify({
            type: "pose-webrtc-stats",
            pc_state: pc.connectionState,
            ice_state: pc.iceConnectionState,
            channel_state: channel.readyState,
            channel_buffered_amount: channel.bufferedAmount,
            selected_pair_id: pair?.id,
            current_rtt_ms: rtt === undefined ? undefined : rtt * 1000,
            available_outgoing_bitrate: statNumber(pair, "availableOutgoingBitrate"),
            pair_bytes_sent: statNumber(pair, "bytesSent"),
            pair_bytes_received: statNumber(pair, "bytesReceived"),
            local_candidate: candidateLabel(local),
            remote_candidate: candidateLabel(remote),
            data_messages_sent: statNumber(dataChannelStats, "messagesSent"),
            data_messages_received: statNumber(dataChannelStats, "messagesReceived"),
            data_bytes_sent: statNumber(dataChannelStats, "bytesSent"),
            data_bytes_received: statNumber(dataChannelStats, "bytesReceived"),
          })
        )
      } catch {
        // Stats are diagnostic only.
      }
    }

    function onMessage(e: MessageEvent) {
      let msg: unknown
      try {
        msg = JSON.parse(e.data as string)
      } catch {
        return
      }
      if (typeof msg !== "object" || msg === null) return
      const m = msg as { type?: string; sdp?: string }
      if (m.type === "pose-webrtc-answer" && typeof m.sdp === "string") {
        pcRef.current?.setRemoteDescription({ type: "answer", sdp: m.sdp }).catch(() => {
          requestedRef.current = false
          closePc()
        })
      } else if (m.type === "pose-webrtc-unavailable") {
        closePc()
      }
    }

    const interval = setInterval(() => {
      const ws = wsRef.current
      if (ws !== attachedWsRef.current) {
        if (attachedWsRef.current && listenerRef.current)
          attachedWsRef.current.removeEventListener("message", listenerRef.current)
        attachedWsRef.current = null
        listenerRef.current = null
        requestedRef.current = false
        closePc()
      }
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      if (!listenerRef.current) {
        listenerRef.current = onMessage
        attachedWsRef.current = ws
        ws.addEventListener("message", onMessage)
      }
      if (!requestedRef.current) {
        requestedRef.current = true
        void start(ws)
      }
    }, POLL_MS)

    const statsInterval = setInterval(() => {
      const ws = attachedWsRef.current
      if (ws) void sendStats(ws)
    }, STATS_MS)

    return () => {
      clearInterval(interval)
      clearInterval(statsInterval)
      detach()
    }
  }, [wsRef])

  return channelRef
}
