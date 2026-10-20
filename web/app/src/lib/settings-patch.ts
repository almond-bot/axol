import type { SettingValue } from "./supervisor"

/** Diff against the values present when editing began.
 *
 * A newer server snapshot may contain settings saved by another caller while
 * this form is open. Comparing an old draft with that newer snapshot would
 * mistake those new keys for local resets and delete them on Save.
 */
export function diffSettingValues(
  base: Record<string, SettingValue>,
  draft: Record<string, SettingValue>,
  excluded: ReadonlySet<string>
): Record<string, SettingValue | null> {
  const patch: Record<string, SettingValue | null> = {}
  for (const [key, value] of Object.entries(draft)) {
    if (excluded.has(key)) continue
    if (JSON.stringify(base[key]) !== JSON.stringify(value)) patch[key] = value
  }
  for (const key of Object.keys(base)) {
    if (excluded.has(key)) continue
    if (!(key in draft)) patch[key] = null
  }
  return patch
}

/** Move only successfully saved keys into the draft's comparison base. */
export function applySavedSettingValues(
  base: Record<string, SettingValue>,
  saved: Record<string, SettingValue | null>
): Record<string, SettingValue> {
  const next = { ...base }
  for (const [key, value] of Object.entries(saved)) {
    if (value === null) delete next[key]
    else next[key] = value
  }
  return next
}
