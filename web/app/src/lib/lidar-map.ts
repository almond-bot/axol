/**
 * The live map on the Sensors page: map points, where the robot has been and
 * where it is now, from `/api/lidar/ws?map=1` (server side:
 * `almond_axol/serve/lidar_map.py`). Kept free of three.js so it tests
 * without a renderer.
 */
import { heightColor } from "@/lib/lidar"

/** What the map is doing: building (`mapping`), or finding the robot on a saved map. */
export type LidarMapState =
  | "starting"
  | "mapping"
  | "initializing"
  | "ok"
  | "weak"
  | "lost"
  | "error"

export interface LidarMapStatus {
  /** The saved map shown, or null for a live map. */
  source: string | null
  epoch: number
  error: string | null
  state: LidarMapState
  /** Points in the map. */
  points: number
  /** Metres driven since the map started. */
  distance: number
  /** The map reached its size limit and stopped growing. */
  full: boolean
}

const STATES = new Set<LidarMapState>([
  "starting",
  "mapping",
  "initializing",
  "ok",
  "weak",
  "lost",
  "error",
])

export function parseMapStatus(raw: unknown): LidarMapStatus | null {
  if (!raw || typeof raw !== "object") return null
  const m = raw as Record<string, unknown>
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : 0)
  return {
    source: typeof m.source === "string" && m.source ? m.source : null,
    epoch: num(m.epoch),
    error: typeof m.error === "string" && m.error ? m.error : null,
    state: STATES.has(m.state as LidarMapState) ? (m.state as LidarMapState) : "starting",
    points: num(m.points),
    distance: num(m.distance),
    full: m.full === true,
  }
}

/** One status line for the map, and how it should look. */
export function mapStatusLine(status: LidarMapStatus | null): {
  text: string
  tone: "ok" | "wait" | "bad"
} {
  if (!status) return { text: "Waiting for the lidar…", tone: "wait" }
  if (status.error) return { text: status.error, tone: "bad" }
  const size =
    status.points >= 1000
      ? `${Math.round(status.points / 1000)}k points`
      : `${status.points} points`
  switch (status.state) {
    case "mapping":
      return {
        text: `Mapping, ${status.distance.toFixed(1)} m driven, ${size}${status.full ? " (full)" : ""}`,
        tone: "ok",
      }
    case "ok":
      return { text: `On the map, ${status.distance.toFixed(1)} m driven`, tone: "ok" }
    case "weak":
      return { text: "On the map, weak match", tone: "wait" }
    case "lost":
      return { text: "Lost. Drive back to a mapped area.", tone: "bad" }
    case "initializing":
      return { text: "Finding the robot on the map…", tone: "wait" }
    case "error":
      return { text: "The map stopped", tone: "bad" }
    default:
      return {
        text: status.source ? "Loading the map…" : "Starting the map…",
        tone: "wait",
      }
  }
}

/** A map update: the whole map (`replace`) or points to add. */
export interface MapChunk {
  epoch: number
  replace: boolean
  /** Metres, map frame (Z up), xyz per point. */
  positions: Float32Array
}

const MAP_MAGIC = 0x314d444c // "LDM1", little-endian
const MAP_HEADER = 16

export function parseMapChunk(buf: ArrayBuffer): MapChunk | null {
  if (buf.byteLength < MAP_HEADER) return null
  const view = new DataView(buf)
  if (view.getUint32(0, true) !== MAP_MAGIC) return null
  const epoch = view.getUint32(4, true)
  const replace = (view.getUint32(8, true) & 1) === 1
  const n = view.getUint32(12, true)
  if (buf.byteLength < MAP_HEADER + n * 6) return null
  const cm = new Int16Array(buf.slice(MAP_HEADER, MAP_HEADER + n * 6))
  const positions = new Float32Array(n * 3)
  for (let i = 0; i < positions.length; i++) positions[i] = cm[i] / 100
  return { epoch, replace, positions }
}

/** True if a binary message is a map chunk (rather than a scan). */
export function isMapChunk(buf: ArrayBuffer): boolean {
  return buf.byteLength >= 4 && new DataView(buf).getUint32(0, true) === MAP_MAGIC
}

/** The robot's pose on the map. */
export interface MapPose {
  epoch: number
  /** The first three rows of T_map_base, row by row. */
  m: number[]
  state: LidarMapState
  match: number | null
  /** The scan this pose is for (its `seq`), when the server says. */
  seq: number | null
}

export function parseMapPose(raw: unknown): MapPose | null {
  if (!raw || typeof raw !== "object") return null
  const msg = raw as Record<string, unknown>
  if (msg.type !== "pose" || !Array.isArray(msg.m) || msg.m.length !== 12) return null
  if (!msg.m.every((v) => typeof v === "number" && Number.isFinite(v))) return null
  return {
    epoch: typeof msg.epoch === "number" ? msg.epoch : 0,
    m: msg.m as number[],
    state: STATES.has(msg.state as LidarMapState) ? (msg.state as LidarMapState) : "mapping",
    match: typeof msg.match === "number" ? msg.match : null,
    seq: typeof msg.seq === "number" ? msg.seq : null,
  }
}

export function parseMapTrail(raw: unknown): { epoch: number; points: number[] } | null {
  if (!raw || typeof raw !== "object") return null
  const msg = raw as Record<string, unknown>
  if (msg.type !== "trail" || !Array.isArray(msg.points)) return null
  const points: number[] = []
  for (const p of msg.points) {
    if (Array.isArray(p) && p.length === 3 && p.every((v) => typeof v === "number")) {
      points.push(p[0], p[1], p[2])
    }
  }
  return { epoch: typeof msg.epoch === "number" ? msg.epoch : 0, points }
}

const TRAIL_STEP = 0.1

/**
 * The map, trail and pose as they arrive. Buffers grow in place; `revision`
 * ticks on any change and `generation` when the map was replaced, so a
 * renderer knows whether to upload only the new points.
 */
export class LidarMap {
  epoch = -1
  positions = new Float32Array(0)
  colors = new Float32Array(0)
  count = 0
  /** Ticks on every change. */
  revision = 0
  /** Ticks when the buffers or the whole map were replaced. */
  generation = 0
  trail: number[] = []
  pose: MapPose | null = null
  private listeners = new Set<() => void>()

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  private changed() {
    this.revision++
    for (const fn of this.listeners) fn()
  }

  private reserve(total: number) {
    if (total <= this.positions.length / 3) return
    let cap = Math.max(65536, this.positions.length / 3)
    while (cap < total) cap *= 2
    const positions = new Float32Array(cap * 3)
    const colors = new Float32Array(cap * 3)
    positions.set(this.positions.subarray(0, this.count * 3))
    colors.set(this.colors.subarray(0, this.count * 3))
    this.positions = positions
    this.colors = colors
    // New buffers: the renderer uploads everything again.
    this.generation++
  }

  /** Apply a map update. Updates for an older map are ignored. */
  apply(chunk: MapChunk) {
    if (chunk.replace) {
      this.epoch = chunk.epoch
      this.count = 0
      this.trail = []
      this.pose = null
      this.generation++
    } else if (chunk.epoch !== this.epoch) {
      return
    }
    const n = chunk.positions.length / 3
    this.reserve(this.count + n)
    this.positions.set(chunk.positions, this.count * 3)
    for (let i = 0; i < n; i++) {
      const at = (this.count + i) * 3
      heightColor(chunk.positions[i * 3 + 2], this.colors, at)
    }
    this.count += n
    this.changed()
  }

  setTrail(epoch: number, points: number[]) {
    if (epoch !== this.epoch) return
    this.trail = points
    this.changed()
  }

  setPose(pose: MapPose) {
    if (pose.epoch !== this.epoch) return
    this.pose = pose
    const [x, y, z] = [pose.m[3], pose.m[7], pose.m[11]]
    const t = this.trail
    const n = t.length
    if (n < 3 || Math.hypot(x - t[n - 3], y - t[n - 2]) >= TRAIL_STEP) t.push(x, y, z)
    this.changed()
  }

  clear() {
    this.epoch = -1
    this.count = 0
    this.trail = []
    this.pose = null
    this.generation++
    this.changed()
  }
}

/** The saved maps the host offers (`axol nav.map`). */
export async function fetchSavedMaps(url: string): Promise<string[]> {
  try {
    const r = await fetch(url)
    if (!r.ok) return []
    const body = (await r.json()) as { maps?: unknown }
    return Array.isArray(body.maps) ? body.maps.filter((m) => typeof m === "string") : []
  } catch {
    return []
  }
}
