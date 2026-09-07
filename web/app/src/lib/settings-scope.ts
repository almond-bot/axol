/**
 * How the shared settings are split on the control panel.
 *
 * Settings are grouped by what they configure: the Axol arms, the handheld
 * Mantis rigs, or the host itself (everything shared). Each connection tile
 * opens its own scope, and the scope switcher in the settings card header
 * moves between them. A tab is either a fixed panel key or a settings-schema
 * category key; the scope is implied by the tab, so one piece of state is
 * enough to open the right device's settings from anywhere.
 */
export type SettingsScope = "axol" | "mantis" | "general"

export type SettingsTab = string

export const SETTINGS_SCOPES: { key: SettingsScope; label: string; title: string }[] = [
  { key: "axol", label: "Axol", title: "Axol settings" },
  { key: "mantis", label: "Mantis", title: "Mantis settings" },
  { key: "general", label: "General", title: "General settings" },
]

/** Schema categories that configure the Axol arms; every other category (and
 *  any a newer host adds) is a host-wide setting under General. */
export const AXOL_CATEGORY_KEYS = new Set(["robot", "teleop", "kinematics"])
const AXOL_TABS = new Set(["cameras", "pose"])
const MANTIS_TABS = new Set(["mantis-tracking", "mantis-can", "mantis-cameras"])

export function settingsScopeForTab(tab: SettingsTab): SettingsScope {
  if (AXOL_TABS.has(tab) || AXOL_CATEGORY_KEYS.has(tab)) return "axol"
  if (MANTIS_TABS.has(tab)) return "mantis"
  return "general"
}

/** The tab a scope opens on when entered from its connection tile. */
export function defaultSettingsTab(scope: SettingsScope): SettingsTab {
  return scope === "axol" ? "cameras" : scope === "mantis" ? "mantis-tracking" : "usb"
}
