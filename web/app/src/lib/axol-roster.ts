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
    kind: "Axol only",
    pitch: "Shoulders, head camera and arms. Bolt it where you need it.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  ox_cart: {
    model: "ox_cart",
    name: "Axol × Ox",
    kind: "Lift cart",
    pitch: "Axol on a rolling cart with a 457 mm lift.",
    stats: [
      { label: "Lift", value: 457 / LIFT_MAX_MM, reading: "457 mm" },
      { label: "Mobility", value: 0.55, reading: "Rolls" },
    ],
  },
  owl: {
    model: "owl",
    name: "Axol × Owl",
    kind: "Column mount",
    pitch: "Axol on a compact column and foot plate, for a fixed station.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  mobile: {
    model: "mobile",
    name: "Axol × Jelly",
    kind: "Mobile base",
    pitch:
      "Axol on Jelly, a powered mobile base with a 680 mm two-stage lift. Axol uses this model whenever Jelly is attached.",
    stats: [
      { label: "Lift", value: 1, reading: "680 mm" },
      { label: "Mobility", value: 1, reading: "Drives" },
    ],
  },
}
