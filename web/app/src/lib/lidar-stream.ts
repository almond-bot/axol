import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { apiUrl, wsBaseUrl } from "@/lib/supervisor"
import { LidarCloud, parseLidarFrame, parseLidarStatus, type LidarStatus } from "@/lib/lidar"
import {
  isMapChunk,
  LidarMap,
  parseMapChunk,
  parseMapPose,
  parseMapStatus,
  parseMapTrail,
  type LidarMapStatus,
} from "@/lib/lidar-map"

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

/** A request to change the shared live map. */
export type LidarMapCommand = { type: "map-source"; source: string | null } | { type: "reset-map" }

export interface LidarStreamResult {
  cloud: LidarCloud
  status: LidarStatus
  /** The live map (only with `map: true`). */
  map: LidarMap
  mapStatus: LidarMapStatus | null
  send: (command: LidarMapCommand) => void
}

/**
 * Live lidar frames from the host while `enabled`. Frames go into the
 * returned `cloud` (read it from a render loop); `status` re-renders only
 * when the server's status changes. With `map`, the live map and the
 * robot's pose on it go into `map` too.
 */
export function useLidarStream(
  enabled: boolean,
  points = 12000,
  { map: withMap = false }: { map?: boolean } = {}
): LidarStreamResult {
  const cloud = useMemo(() => new LidarCloud(), [])
  const map = useMemo(() => new LidarMap(), [])
  const [status, setStatus] = useState<LidarStatus>(CONNECTING)
  const [mapStatus, setMapStatus] = useState<LidarMapStatus | null>(null)
  const socket = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      if (closed) return
      ws = new WebSocket(`${wsBaseUrl()}/api/lidar/ws?points=${points}${withMap ? "&map=1" : ""}`)
      ws.binaryType = "arraybuffer"
      const open = ws
      ws.onopen = () => {
        socket.current = open
      }
      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          if (isMapChunk(event.data)) {
            const chunk = parseMapChunk(event.data)
            if (chunk) map.apply(chunk)
            return
          }
          const frame = parseLidarFrame(event.data, performance.now())
          if (frame) cloud.add(frame)
          return
        }
        try {
          const raw = JSON.parse(event.data as string) as Record<string, unknown>
          if (raw.type === "pose") {
            const pose = parseMapPose(raw)
            if (pose) {
              map.setPose(pose)
              if (pose.seq !== null) cloud.setPose(pose.seq, pose.epoch, pose.m)
            }
            return
          }
          if (raw.type === "trail") {
            const trail = parseMapTrail(raw)
            if (trail) map.setTrail(trail.epoch, trail.points)
            return
          }
          if (withMap && raw.type === "status") {
            const next = parseMapStatus(raw.map)
            setMapStatus((prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next))
          }
          const next = parseLidarStatus(raw)
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
        socket.current = null
        setStatus(CONNECTING)
        if (!closed) retry = setTimeout(connect, RECONNECT_MS)
      }
      ws.onerror = () => ws?.close()
    }
    connect()

    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      socket.current = null
      if (ws) {
        ws.onclose = null
        ws.close()
      }
      cloud.clear()
      map.clear()
      setStatus(CONNECTING)
      setMapStatus(null)
    }
  }, [enabled, points, withMap, cloud, map])

  const send = useCallback((command: LidarMapCommand) => {
    const ws = socket.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(command))
  }, [])

  return { cloud, status, map, mapStatus, send }
}
