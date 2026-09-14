/**
 * The on-disk shape of the robot's shared settings (~/.almond/settings.json,
 * serve/settings.py) for the panel's Export / Import buttons.
 *
 * The API exchanges values flat, keyed by canonical dotted path
 * ("axol.left.elbow.kp"); the file nests the same keys into one object per
 * section so it reads like the robot's config — and so a teleop
 * `--config_path` file and the `axol` / `teleop` / `kinematics` sections
 * share one vocabulary. Export writes that file; Import accepts it, or the
 * pre-v2 layout ({ values, advanced, cameras }) an older panel exported (the
 * server maps its old key names onto the canonical ones).
 */

export const SETTINGS_FILE_VERSION = 2

/** Top-level keys of the file that are not settings sections. */
const RESERVED_KEYS = new Set(["version", "cameras"])

export type SettingsLeaf = string | number | boolean | (number | string)[]
export type SettingsTree = { [key: string]: SettingsLeaf | SettingsTree }

export interface SettingsFile {
  version: number
  cameras: unknown
  [section: string]: unknown
}

function isTree(value: unknown): value is SettingsTree {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** `{ axol: { left: { kp: 1 } } }` → `{ "axol.left.kp": 1 }`; arrays are leaves. */
export function flattenSettings(tree: SettingsTree, prefix = ""): Record<string, SettingsLeaf> {
  const flat: Record<string, SettingsLeaf> = {}
  for (const [name, value] of Object.entries(tree)) {
    const key = `${prefix}${name}`
    if (isTree(value)) Object.assign(flat, flattenSettings(value, `${key}.`))
    else flat[key] = value
  }
  return flat
}

/** Inverse of {@link flattenSettings}: keys sorted so the output is stable. */
export function nestSettings(flat: Record<string, SettingsLeaf>): SettingsTree {
  const tree: SettingsTree = {}
  for (const key of Object.keys(flat).sort()) {
    const parts = key.split(".")
    const leaf = parts.pop()!
    let node = tree
    for (const part of parts) {
      const child = node[part]
      if (isTree(child)) node = child
      else node = node[part] = {}
    }
    node[leaf] = flat[key]
  }
  return tree
}

/** The settings file for the given stored values and camera spec. */
export function buildSettingsFile(
  values: Record<string, SettingsLeaf>,
  cameras: unknown
): SettingsFile {
  return { version: SETTINGS_FILE_VERSION, ...nestSettings(values), cameras }
}

/**
 * Read a settings file (either layout) back into flat values and cameras.
 * Keys are returned as written; the server canonicalizes pre-v2 names.
 */
export function parseSettingsFile(data: unknown): {
  values: Record<string, SettingsLeaf>
  cameras: unknown
} {
  if (!isTree(data)) throw new Error("invalid settings file")
  if ("values" in data || "advanced" in data) {
    const values = isTree(data.values) ? data.values : {}
    const advanced = isTree(data.advanced) ? data.advanced : {}
    return {
      values: {
        ...(advanced as Record<string, SettingsLeaf>),
        ...(values as Record<string, SettingsLeaf>),
      },
      cameras: data.cameras,
    }
  }
  const sections: SettingsTree = {}
  for (const [key, value] of Object.entries(data)) {
    if (!RESERVED_KEYS.has(key) && isTree(value)) sections[key] = value
  }
  return { values: flattenSettings(sections), cameras: data.cameras }
}
