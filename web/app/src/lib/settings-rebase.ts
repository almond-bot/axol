import type { SettingValue } from "@/lib/supervisor"

/** ``draft`` moved onto new stored ``next`` values: every key whose draft
 *  value still equals what it was seeded from (``seed``) takes ``next``'s;
 *  keys the operator changed keep their draft value. */
export function rebaseValues(
  draft: Record<string, SettingValue>,
  seed: Record<string, SettingValue>,
  next: Record<string, SettingValue>
): Record<string, SettingValue> {
  const out = { ...draft }
  for (const key of new Set([...Object.keys(seed), ...Object.keys(next)])) {
    if (JSON.stringify(draft[key]) !== JSON.stringify(seed[key])) continue
    if (key in next) out[key] = next[key]
    else delete out[key]
  }
  return out
}
