/**
 * Live lidar scans from `axol serve` (`/api/lidar/ws`, see
 * `almond_axol/serve/lidar_link.py`), kept free of three.js so it tests
 * without a renderer.
 *
 * Points arrive in Jelly's base frame: metres, Z up, origin on the floor
 * under the centre of the base, +X forward.
 */

export type LidarColorMode = "height" | "reflectivity"

export interface LidarSettings {
  /** Seconds of scans to show at once. */
  history: number
  /** Point size in centimetres. */
  pointSize: number
  /** Hide points higher than this (m above the floor), to see under a ceiling. */
  hideAbove: number
  colorBy: LidarColorMode
}

export const DEFAULT_LIDAR_SETTINGS: LidarSettings = {
  history: 1,
  pointSize: 1.5,
  hideAbove: 2.2,
  colorBy: "height",
}

export interface LidarMount {
  x: number
  y: number
  z: number
  roll: number
  pitch: number
  yaw: number
}

export interface LidarStatus {
  state: "connecting" | "starting" | "streaming" | "error"
  error: string | null
  pointsPerSecond: number
  mount: LidarMount | null
}

/** One decoded frame: positions in metres and 0..255 reflectivity. */
export interface LidarFrame {
  seq: number
  count: number
  positions: Float32Array
  reflectivity: Uint8Array
  /** `performance.now()` when it arrived (ms). */
  receivedAt: number
  /** Where the robot was on the live map for this scan (see `LidarCloud.setPose`). */
  pose?: { epoch: number; m: number[] }
}

const MAGIC = 0x3152444c // "LDR1" little-endian
const HEADER = 12

/** Decode one binary frame message, or null when it isn't one. */
export function parseLidarFrame(buf: ArrayBuffer, receivedAt = 0): LidarFrame | null {
  if (buf.byteLength < HEADER) return null
  const view = new DataView(buf)
  if (view.getUint32(0, true) !== MAGIC) return null
  const seq = view.getUint32(4, true)
  const count = view.getUint32(8, true)
  if (buf.byteLength < HEADER + count * 7) return null
  const mm = new Int16Array(buf, HEADER, count * 3)
  const positions = new Float32Array(count * 3)
  for (let i = 0; i < mm.length; i++) positions[i] = mm[i] / 1000
  const reflectivity = new Uint8Array(buf, HEADER + count * 6, count)
  return { seq, count, positions, reflectivity, receivedAt }
}

/** Status fields from the server's JSON message. */
export function parseLidarStatus(msg: unknown): LidarStatus | null {
  if (!msg || typeof msg !== "object") return null
  const m = msg as Record<string, unknown>
  if (m.type !== "status") return null
  const state = m.state
  const mount = m.mount as Partial<LidarMount> | undefined
  return {
    state: state === "streaming" || state === "error" || state === "starting" ? state : "starting",
    error: typeof m.error === "string" && m.error ? m.error : null,
    pointsPerSecond: typeof m.pointsPerSecond === "number" ? m.pointsPerSecond : 0,
    mount:
      mount && typeof mount.z === "number"
        ? {
            x: mount.x ?? 0,
            y: mount.y ?? 0,
            z: mount.z,
            roll: mount.roll ?? 0,
            pitch: mount.pitch ?? 0,
            yaw: mount.yaw ?? 0,
          }
        : null,
  }
}

/** A short line for a view's header, and how it should look. */
export function lidarStatusLine(status: LidarStatus): {
  text: string
  tone: "ok" | "wait" | "bad"
} {
  if (status.state === "streaming") {
    const k = Math.round(status.pointsPerSecond / 1000)
    return { text: k > 0 ? `Live, ${k}k points/s` : "Live", tone: "ok" }
  }
  if (status.state === "error") return { text: status.error ?? "No scan", tone: "bad" }
  if (status.state === "starting") return { text: "Starting the lidar…", tone: "wait" }
  return { text: "Connecting…", tone: "wait" }
}

// three.js treats vertex colors as linear, so the colors below (sRGB) are
// converted once here.
const toLinear = (c: number) => Math.pow(c, 2.2)

// Height colors from the floor (0 m) up to 2.5 m: blue, cyan, green, yellow, red.
const RAMP: [number, number, number][] = (
  [
    [0.19, 0.3, 0.86],
    [0.1, 0.75, 0.9],
    [0.25, 0.85, 0.35],
    [0.98, 0.85, 0.2],
    [0.95, 0.3, 0.2],
  ] as [number, number, number][]
).map((c) => c.map(toLinear) as [number, number, number])
const RAMP_TOP = 2.5

/** Linear RGB (0..1) for a height in metres. */
export function heightColor(z: number, out: Float32Array, i: number) {
  const t = Math.min(Math.max(z / RAMP_TOP, 0), 1) * (RAMP.length - 1)
  const k = Math.min(Math.floor(t), RAMP.length - 2)
  const f = t - k
  const a = RAMP[k]
  const b = RAMP[k + 1]
  out[i] = a[0] + (b[0] - a[0]) * f
  out[i + 1] = a[1] + (b[1] - a[1]) * f
  out[i + 2] = a[2] + (b[2] - a[2]) * f
}

/** Linear RGB (0..1) for a reflectivity: dark grey for dull, white for shiny. */
export function reflectivityColor(r: number, out: Float32Array, i: number) {
  // Most surfaces return under 100; stretch that range.
  const v = 0.25 + 0.75 * Math.min(Math.sqrt(r / 150), 1)
  out[i] = toLinear(v)
  out[i + 1] = toLinear(v * 0.97)
  out[i + 2] = toLinear(v * 0.9)
}

/**
 * The scans received in the last few seconds, merged into one point buffer
 * for drawing. `fill` reuses the output arrays, growing them when needed.
 */
export class LidarCloud {
  private frames: LidarFrame[] = []
  /** Bumped whenever a frame arrives or is cleared. */
  revision = 0
  /** Longest history any view asks for (s); older frames are dropped. */
  keep = 10
  private listeners = new Set<() => void>()

  add(frame: LidarFrame) {
    this.frames.push(frame)
    const cutoff = frame.receivedAt - this.keep * 1000
    while (this.frames.length > 1 && this.frames[0].receivedAt < cutoff) this.frames.shift()
    this.changed()
  }

  clear() {
    this.frames = []
    this.changed()
  }

  /**
   * The robot's pose on the live map when scan `seq` was taken: the first
   * three rows of T_map_base. Lets `fill` draw scans on the map.
   */
  setPose(seq: number, epoch: number, m: number[]) {
    for (let i = this.frames.length - 1; i >= 0; i--) {
      const f = this.frames[i]
      if (f.seq === seq) {
        f.pose = { epoch, m }
        this.changed()
        return
      }
      if (f.seq < seq) return
    }
  }

  /** The pose for frame `i`: its own, or the nearest posed frame's. */
  private poseFor(i: number, epoch: number): number[] | null {
    const own = this.frames[i].pose
    if (own?.epoch === epoch) return own.m
    for (let d = 1; d < this.frames.length; d++) {
      for (const j of [i - d, i + d]) {
        const p = this.frames[j]?.pose
        if (p?.epoch === epoch) return p.m
      }
    }
    return null
  }

  /** Call `listener` whenever the frames change; returns the unsubscribe. */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  private changed() {
    this.revision++
    for (const listener of this.listeners) listener()
  }

  get size() {
    return this.frames.length
  }

  /**
   * Merge the frames newer than `history` seconds before the newest one
   * into `positions` and `colors`, leaving out points above `hideAbove`.
   * Returns the arrays (new ones when the old were too small) and the count.
   *
   * With `mapEpoch`, points come out in that live map's frame instead of the
   * base frame, each scan placed where the robot was when it was taken, so
   * the history doesn't smear as the robot turns. Scans without a pose on
   * that map are left out.
   */
  fill(
    settings: LidarSettings,
    positions: Float32Array,
    colors: Float32Array,
    mapEpoch: number | null = null
  ): { positions: Float32Array; colors: Float32Array; count: number } {
    const newest = this.frames[this.frames.length - 1]
    if (!newest) return { positions, colors, count: 0 }
    // Always keep the newest frame, however short the history.
    const cutoff = newest.receivedAt - settings.history * 1000
    const shown: { f: LidarFrame; m: number[] | null }[] = []
    this.frames.forEach((f, i) => {
      if (f !== newest && f.receivedAt <= cutoff) return
      if (mapEpoch === null) {
        shown.push({ f, m: null })
        return
      }
      const m = this.poseFor(i, mapEpoch)
      if (m) shown.push({ f, m })
    })
    const total = shown.reduce((n, s) => n + s.f.count, 0)
    if (positions.length < total * 3) {
      positions = new Float32Array(total * 3)
      colors = new Float32Array(total * 3)
    }
    let n = 0
    for (const { f, m } of shown) {
      const p = f.positions
      for (let i = 0; i < f.count; i++) {
        let x = p[i * 3]
        let y = p[i * 3 + 1]
        let z = p[i * 3 + 2]
        if (m) {
          const bx = x
          const by = y
          const bz = z
          x = m[0] * bx + m[1] * by + m[2] * bz + m[3]
          y = m[4] * bx + m[5] * by + m[6] * bz + m[7]
          z = m[8] * bx + m[9] * by + m[10] * bz + m[11]
        }
        if (z > settings.hideAbove) continue
        const o = n * 3
        positions[o] = x
        positions[o + 1] = y
        positions[o + 2] = z
        if (settings.colorBy === "height") heightColor(z, colors, o)
        else reflectivityColor(f.reflectivity[i], colors, o)
        n++
      }
    }
    return { positions, colors, count: n }
  }
}

const SETTINGS_KEY = "axol.lidarView"

/** The viewer's saved settings, or the defaults. */
export function loadLidarSettings(): LidarSettings {
  try {
    const raw = globalThis.localStorage?.getItem(SETTINGS_KEY)
    if (!raw) return DEFAULT_LIDAR_SETTINGS
    const saved = JSON.parse(raw) as Partial<LidarSettings>
    return sanitizeLidarSettings({ ...DEFAULT_LIDAR_SETTINGS, ...saved })
  } catch {
    return DEFAULT_LIDAR_SETTINGS
  }
}

export function saveLidarSettings(settings: LidarSettings) {
  try {
    globalThis.localStorage?.setItem(SETTINGS_KEY, JSON.stringify(settings))
  } catch {
    // per-viewer convenience only
  }
}

const clamp = (v: unknown, lo: number, hi: number, fallback: number) =>
  typeof v === "number" && Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : fallback

export function sanitizeLidarSettings(s: Partial<LidarSettings>): LidarSettings {
  const d = DEFAULT_LIDAR_SETTINGS
  return {
    history: clamp(s.history, 0.1, 10, d.history),
    pointSize: clamp(s.pointSize, 0.3, 5, d.pointSize),
    hideAbove: clamp(s.hideAbove, 0.3, 6, d.hideAbove),
    colorBy: s.colorBy === "reflectivity" ? "reflectivity" : "height",
  }
}
