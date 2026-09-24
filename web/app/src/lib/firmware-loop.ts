/**
 * Which motor vendor's firmware loop a joint runs, and which `tune.a4` knobs
 * that loop has.
 *
 * The five MyActuator joints (shoulder_1 … wrist_1) run the 0xA4 cascade:
 * position PID → speed PI → current PI, plus a stored planner acceleration.
 * The two Damiao wrists run the position-velocity mode: position PI (KP_APR /
 * KI_APR) → speed PI (KP_ASR / KI_ASR) under the ACC/DEC profiler — no
 * position D, no exposed current loop, no planner. `tune.a4` refuses a knob
 * the selected joint's vendor does not have, so the form hides it.
 */

import { MYACTUATOR_JOINTS } from "./wire-mode"

export type FirmwareVendor = "myactuator" | "damiao"

/** The Damiao wrists, the only arm joints not on a MyActuator motor. */
export const DAMIAO_JOINTS = ["wrist_2", "wrist_3"]

/** The vendor behind an arm joint, or null for no (or an unknown) joint. */
export function jointVendor(joint: string | undefined): FirmwareVendor | null {
  if (!joint) return null
  if (MYACTUATOR_JOINTS.includes(joint)) return "myactuator"
  if (DAMIAO_JOINTS.includes(joint)) return "damiao"
  return null
}

/**
 * Whether a field restricted to `vendors` shows for `joint`. Unrestricted
 * fields always show; with no joint picked yet every field shows, so the
 * form does not jump around before the choice that decides it.
 */
export function shownForJoint(
  vendors: readonly FirmwareVendor[] | undefined,
  joint: string | undefined
): boolean {
  const vendor = jointVendor(joint)
  return !vendors || vendor == null || vendors.includes(vendor)
}
