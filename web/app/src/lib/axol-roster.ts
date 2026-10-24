/**
 * Every Axol version as the robot-model pickers present it: name, the line
 * under it, the pitch, and the stage's stats.
 */

import type { AxolModel } from "@/lib/supervisor"

export interface Character {
  model: AxolModel
  name: string
  /** The short line under the name on the roster card. */
  kind: string
  pitch: string
  /** Each stat: a 0..1 bar and what the bar means. */
  stats: { label: string; value: number; reading: string }[]
}

const LIFT_MAX_MM = 680

export const ROSTER: Record<AxolModel, Character> = {
  classic: {
    model: "classic",
    name: "Axol Legacy",
    kind: "Original model",
    pitch: "The original Axol model, and the default.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  standalone: {
    model: "standalone",
    name: "Axol Standalone",
    kind: "No mount",
    pitch: "Axol on its own, bolted to your table or fixture.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  ox_cart: {
    model: "ox_cart",
    name: "Axol on Ox Cart",
    kind: "Lift cart",
    pitch: "Axol on a height-adjustable cart with leveling casters.",
    stats: [
      { label: "Lift", value: 457 / LIFT_MAX_MM, reading: "457 mm" },
      { label: "Mobility", value: 0.55, reading: "Rolls" },
    ],
  },
  owl: {
    model: "owl",
    name: "Axol on Owl Mount",
    kind: "Table column",
    pitch: "Axol on a T-slot column that bolts to a table or fixture.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  mobile: {
    model: "mobile",
    name: "Axol on Jelly",
    kind: "Mobile base",
    pitch: "Axol on a powered mobile base with a lift. Used automatically when Jelly is attached.",
    stats: [
      { label: "Lift", value: 1, reading: "680 mm" },
      { label: "Mobility", value: 1, reading: "Drives" },
    ],
  },
}
