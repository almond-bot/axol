/**
 * How the shared settings are split on the control panel.
 *
 * Settings are grouped by what they configure: the Axol arms, the handheld
 * Mantis rigs, Jelly (the powered base and lift the Axol rides on), or the
 * host itself (everything shared). Each connection tile opens its own scope,
 * and the scope switcher in the settings card header moves between them. A
 * tab is either a fixed panel key or a settings-schema category key; the
 * scope is implied by the tab, so one piece of state is enough to open the
 * right device's settings from anywhere.
 */
export type SettingsScope = "axol" | "mantis" | "jelly" | "general"

export type SettingsTab = string

export const SETTINGS_SCOPES: { key: SettingsScope; label: string; title: string }[] = [
  { key: "axol", label: "Axol", title: "Axol settings" },
  { key: "mantis", label: "Mantis", title: "Mantis settings" },
  { key: "jelly", label: "Jelly", title: "Jelly settings" },
  { key: "general", label: "General", title: "General settings" },
]

/** Schema categories that configure the Axol arms; every other category (and
 *  any a newer host adds) is a host-wide setting under General. */
export const AXOL_CATEGORY_KEYS = new Set(["robot", "teleop", "kinematics"])
/** The Jelly switches (wheels / lift) — the Jelly scope's first tab. */
export const JELLY_CATEGORY_KEYS = new Set(["jelly"])
/** The Jelly config tree (speeds, slew, heading hold) rendered as its own
 *  tab instead of under General → Advanced. */
export const JELLY_PARAMETERS_TAB = "jelly-parameters"
/** The Advanced section key that backs the Jelly → Parameters tab. */
export const JELLY_ADVANCED_SECTION = "jelly"
const AXOL_TABS = new Set(["cameras", "pose"])
const MANTIS_TABS = new Set(["mantis-tracking", "mantis-can", "mantis-cameras"])
const JELLY_TABS = new Set([JELLY_PARAMETERS_TAB])

export function settingsScopeForTab(tab: SettingsTab): SettingsScope {
  if (AXOL_TABS.has(tab) || AXOL_CATEGORY_KEYS.has(tab)) return "axol"
  if (MANTIS_TABS.has(tab)) return "mantis"
  if (JELLY_TABS.has(tab) || JELLY_CATEGORY_KEYS.has(tab)) return "jelly"
  return "general"
}

/** The tab a scope opens on when entered from its connection tile. */
export function defaultSettingsTab(scope: SettingsScope): SettingsTab {
  switch (scope) {
    case "axol":
      return "cameras"
    case "mantis":
      return "mantis-tracking"
    case "jelly":
      return "jelly"
    default:
      return "usb"
  }
}
