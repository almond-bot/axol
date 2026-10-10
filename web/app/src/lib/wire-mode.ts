/**
 * Per-joint controller split for `tune.motion --a4`.
 *
 * A reference-motion replay drives every joint on the MIT impedance frame
 * (the production law) unless the joint is named with `--a4 SIDE.JOINT`,
 * which hands it to the motor's own firmware position loop (0xA4) for that
 * run — the way to A/B "shoulder_1 on the firmware loop, everything else on
 * impedance" against the identical motion. The flag is repeatable and the
 * server turns a whitespace-separated field into one `--a4` per token, so
 * the form value is the token string the CLI takes: `right.shoulder_1
 * left.shoulder_1`. Only MyActuator joints have a firmware loop to hand
 * over to; the Damiao wrists always run MIT.
 */

/** The MyActuator joints, in arm order — the only ones `--a4` accepts usefully. */
export const MYACTUATOR_JOINTS = ["shoulder_1", "shoulder_2", "shoulder_3", "elbow", "wrist_1"]

export const SIDES = ["left", "right"] as const
export type Side = (typeof SIDES)[number]

/** `side.joint` token for one cell of the picker. */
export function a4Token(side: string, joint: string): string {
  return `${side}.${joint}`
}

/** The `side.joint` tokens set in a form value (unknown tokens are dropped). */
export function parseA4Tokens(value: string | undefined): Set<string> {
  const out = new Set<string>()
  for (const tok of (value ?? "").split(/\s+/).filter(Boolean)) {
    const [side = "", joint = ""] = tok.split(".")
    if ((SIDES as readonly string[]).includes(side) && MYACTUATOR_JOINTS.includes(joint)) {
      out.add(a4Token(side, joint))
    }
  }
  return out
}

/** Serialize back to the token string, in a stable side-major order. */
export function serializeA4Tokens(tokens: Iterable<string>): string {
  const have = new Set(tokens)
  const out: string[] = []
  for (const side of SIDES) {
    for (const joint of MYACTUATOR_JOINTS) {
      const t = a4Token(side, joint)
      if (have.has(t)) out.push(t)
    }
  }
  return out.join(" ")
}

/** Flip one cell's controller and return the new form value. */
export function toggleA4Token(value: string | undefined, side: string, joint: string): string {
  const tokens = parseA4Tokens(value)
  const t = a4Token(side, joint)
  if (tokens.has(t)) tokens.delete(t)
  else tokens.add(t)
  return serializeA4Tokens(tokens)
}

/**
 * The controller a joint runs on for this run: `a4` when the form names it
 * or the robot's config already sets `wire_mode a4` (a run can add `--a4`
 * joints on top of the config but not take one away), else `mit`.
 */
export function effectiveWireMode(
  value: string | undefined,
  configModes: Record<string, Record<string, string>> | null | undefined,
  side: string,
  joint: string
): "mit" | "a4" {
  if (parseA4Tokens(value).has(a4Token(side, joint))) return "a4"
  return (configModes?.[side]?.[joint] ?? "mit").toLowerCase() === "a4" ? "a4" : "mit"
}
