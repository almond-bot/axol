/**
 * This browser's copy of a host's camera spec. The host's shared settings are
 * the source of truth; the copy only fills the panel while they load, and
 * stands in for them on a host too old to store cameras itself.
 *
 * The copy is kept per host. One browser often talks to several robots (an
 * engineer helping a customer), and a spec from one robot must never show up,
 * be sent with a run, or be saved on another.
 */
import { cameraSerials, serverHttpBase, type CameraSpec } from "./supervisor"

/** Where browsers before per-host caching kept the one spec they knew. */
export const LEGACY_CAMERA_STORAGE = "axolCameraSpec"
const CAMERA_STORAGE_PREFIX = "axolCameraSpec:"

export const DEFAULT_CAMERAS: CameraSpec = {
  serials: { overhead: "", left_arm: "", right_arm: "" },
  mantis_serials: { overhead: "", left_arm: "", right_arm: "" },
  stream_resolution: "SVGA",
  record_resolution: "SVGA",
  stream: {},
  record: {},
}

/** One key per host origin, so "10.0.0.5" and "https://10.0.0.5:8001" match.
 *  An empty host is the origin that served the page. */
export function cameraStorageKey(host: string): string {
  return `${CAMERA_STORAGE_PREFIX}${serverHttpBase(host)}`
}

/** Parse a stored spec, migrating the older field shapes it may use. */
function parseCameras(raw: string): CameraSpec | null {
  const parsed = JSON.parse(raw)
  if (!parsed || typeof parsed !== "object") return null
  const serials = { ...DEFAULT_CAMERAS.serials, ...(parsed.serials ?? {}) }
  return {
    ...DEFAULT_CAMERAS,
    ...parsed,
    serials,
    mantis_serials: cameraSerials({ serials, mantis_serials: parsed.mantis_serials }, true),
    // Migrate the legacy single `resolution` to the streaming resolution.
    stream_resolution:
      parsed.stream_resolution ?? parsed.resolution ?? DEFAULT_CAMERAS.stream_resolution,
    record_resolution: parsed.record_resolution ?? DEFAULT_CAMERAS.record_resolution,
    // Migrate the earlier per-eye maps to the per-branch participation maps.
    stream: { ...(parsed.stream ?? parsed.stream_eyes ?? {}) },
    record: { ...(parsed.record ?? parsed.record_eyes ?? {}) },
  }
}

/**
 * The spec this browser last saw for `host`, or the defaults.
 *
 * `legacy` also accepts the pre-per-host copy. Only pass it for a host that
 * has no settings store: such a host never held cameras of its own, so the
 * browser-wide copy is what it was always run with.
 */
export function loadCachedCameras(host: string, { legacy = false } = {}): CameraSpec {
  try {
    const raw =
      localStorage.getItem(cameraStorageKey(host)) ??
      (legacy ? localStorage.getItem(LEGACY_CAMERA_STORAGE) : null)
    if (raw) return parseCameras(raw) ?? DEFAULT_CAMERAS
  } catch {
    // ignore malformed or unavailable storage
  }
  return DEFAULT_CAMERAS
}

export function persistCachedCameras(host: string, spec: CameraSpec): void {
  try {
    localStorage.setItem(cameraStorageKey(host), JSON.stringify(spec))
  } catch {
    // ignore storage failures
  }
}
