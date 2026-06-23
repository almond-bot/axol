import { useEffect, useRef, useState } from "react"
import { AxolConnectionStatus } from "./types"

const RETRY_MS = 1000

/**
 * Maintains a dedicated pose WebSocket to the headset's loopback VR server
 * (`wss://localhost:<port>`), reached over the Quest-over-USB `adb reverse`
 * tunnel. Controller poses go over this wired link to avoid the WiFi
 * power-save buffering behind the ~150 ms pose gaps; camera video keeps using
 * the main LAN connection (WebRTC can't cross the TCP port-forward).
 *
 * Auto-connects with retry while `enabled`, and closes when disabled. The
 * returned ref is what `AxolVRClient` sends pose frames to.
 */
export function useAxolPoseSocket(enabled: boolean, port = 8000) {
  const [status, setStatus] = useState<AxolConnectionStatus>(AxolConnectionStatus.Idle)
  const poseWsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) {
      setStatus(AxolConnectionStatus.Idle)
      return
    }

    let cancelled = false
    let retry: ReturnType<typeof setTimeout> | null = null

    function close() {
      const ws = poseWsRef.current
      if (ws) {
        ws.onopen = null
        ws.onclose = null
        ws.onerror = null
        try {
          ws.close()
        } catch {
          // already closed
        }
        poseWsRef.current = null
      }
    }

    function connect() {
      if (cancelled) return
      close()
      setStatus(AxolConnectionStatus.Connecting)

      let ws: WebSocket
      try {
        ws = new WebSocket(`wss://localhost:${port}/ws`)
      } catch {
        setStatus(AxolConnectionStatus.Error)
        retry = setTimeout(connect, RETRY_MS)
        return
      }
      poseWsRef.current = ws

      ws.onopen = () => {
        if (!cancelled) setStatus(AxolConnectionStatus.Open)
      }
      ws.onerror = () => {
        if (!cancelled) setStatus(AxolConnectionStatus.Error)
      }
      ws.onclose = () => {
        if (cancelled) return
        poseWsRef.current = null
        setStatus(AxolConnectionStatus.Connecting)
        retry = setTimeout(connect, RETRY_MS)
      }
    }

    connect()

    return () => {
      cancelled = true
      if (retry) clearTimeout(retry)
      close()
    }
  }, [enabled, port])

  return { poseWsRef, status }
}
