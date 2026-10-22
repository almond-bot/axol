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
    pitch: "The original Axol model. Keep it if this robot was set up before the new mounts.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  standalone: {
    model: "standalone",
    name: "Axol Standalone",
    kind: "Just Axol",
    pitch: "Shoulders, head camera and arms, bolted wherever the work is.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  ox_cart: {
    model: "ox_cart",
    name: "Axol × Ox",
    kind: "Lift cart",
    pitch: "Rides a rolling cart with a 457 mm lift: wheel it to the job, raise it to the work.",
    stats: [
      { label: "Lift", value: 457 / LIFT_MAX_MM, reading: "457 mm" },
      { label: "Mobility", value: 0.55, reading: "Rolls" },
    ],
  },
  owl: {
    model: "owl",
    name: "Axol × Owl",
    kind: "Column mount",
    pitch: "A compact column on a foot plate: a steady, fixed station.",
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
      "The mobile Axol: a self-driving base and a 680 mm two-stage lift. It takes over the moment Jelly is attached.",
    stats: [
      { label: "Lift", value: 1, reading: "680 mm" },
      { label: "Mobility", value: 1, reading: "Drives" },
    ],
  },
}
