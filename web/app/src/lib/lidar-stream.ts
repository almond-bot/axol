import { useEffect, useMemo, useState } from "react"
import { apiUrl, wsBaseUrl } from "@/lib/supervisor"
import { LidarCloud, parseLidarFrame, parseLidarStatus, type LidarStatus } from "@/lib/lidar"

const RECONNECT_MS = 2000

const CONNECTING: LidarStatus = {
  state: "connecting",
  error: null,
  pointsPerSecond: 0,
  mount: null,
}

/**
 * Whether the host has a lidar set up (`axol lidar.setup`). Null while
 * asking, and false when the host can't say (an older backend).
 */
export function useLidarConfigured(): boolean | null {
  const [configured, setConfigured] = useState<boolean | null>(null)
  useEffect(() => {
    let cancelled = false
    fetch(apiUrl("/api/lidar/status"))
      .then((r) => (r.ok ? r.json() : { configured: false }))
      .then((body: { configured?: unknown }) => {
        if (!cancelled) setConfigured(body.configured === true)
      })
      .catch(() => {
        if (!cancelled) setConfigured(false)
      })
    return () => {
      cancelled = true
    }
  }, [])
  return configured
}

/**
 * Live lidar frames from the host while `enabled`. Frames go into the
 * returned `cloud` (read it from a render loop); `status` re-renders only
 * when the server's status changes.
 */
export function useLidarStream(
  enabled: boolean,
  points = 12000
): { cloud: LidarCloud; status: LidarStatus } {
  const cloud = useMemo(() => new LidarCloud(), [])
  const [status, setStatus] = useState<LidarStatus>(CONNECTING)

  useEffect(() => {
    if (!enabled) return
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      if (closed) return
      ws = new WebSocket(`${wsBaseUrl()}/api/lidar/ws?points=${points}`)
      ws.binaryType = "arraybuffer"
      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          const frame = parseLidarFrame(event.data, performance.now())
          if (frame) cloud.add(frame)
          return
        }
        try {
          const next = parseLidarStatus(JSON.parse(event.data as string))
          if (next)
            setStatus((prev) =>
              prev.state === next.state &&
              prev.error === next.error &&
              Math.abs(prev.pointsPerSecond - next.pointsPerSecond) < 5000 &&
              (prev.mount === null) === (next.mount === null)
                ? prev
                : next
            )
        } catch {
          // ignore malformed messages
        }
      }
      ws.onclose = () => {
        setStatus(CONNECTING)
        if (!closed) retry = setTimeout(connect, RECONNECT_MS)
      }
      ws.onerror = () => ws?.close()
    }
    connect()

    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      if (ws) {
        ws.onclose = null
        ws.close()
      }
      cloud.clear()
      setStatus(CONNECTING)
    }
  }, [enabled, points, cloud])

  return { cloud, status }
}
