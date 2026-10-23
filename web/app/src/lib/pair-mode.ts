import type { AxolSettings } from "@almond/axol-vr-client"

// The server's pair mode — one controller leads both arms — under the name it
// has for the fitted gripper: box mode with the parcel gripper (the pair
// aligns on a box, sticks set the width, a stick click flips the grasp) and
// rigid mode with any other (the grippers locked together as they are, moved
// in all six degrees of freedom). The server publishes only one of the two.
export type PairKind = "box" | "rigid"

export function pairKind(settings: AxolSettings | null): PairKind | null {
  if (!settings) return null
  if ("box_mode" in settings.values) return "box"
  if ("rigid_mode" in settings.values) return "rigid"
  return null
}

export function pairKey(kind: PairKind): "box_mode" | "rigid_mode" {
  return kind === "box" ? "box_mode" : "rigid_mode"
}

export function pairModeOn(settings: AxolSettings | null): boolean {
  const kind = pairKind(settings)
  return kind !== null && settings?.values[pairKey(kind)] === true
}

export function pairLabel(kind: PairKind): string {
  return kind === "box" ? "Box" : "Rigid"
}

// Control hints while the pair mode is on, as [control, action] rows.
export function pairHints(kind: PairKind): [string, string][] {
  return kind === "box"
    ? [
        ["Grip", "Lead both arms"],
        ["Stick", "← → width, ↑ ↓ elbows"],
        ["Stick click", "Flush / straight grasp"],
      ]
    : [
        ["Grip", "Lead both arms"],
        ["Stick", "↑ ↓ elbows"],
      ]
}
