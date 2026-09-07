import type { BranchSel, CameraDevice, CameraSlot, CameraSpec, StereoEyes } from "./supervisor"

export const CAMERA_SLOT_KEYS: CameraSlot[] = ["overhead", "left_arm", "right_arm"]

// Default eye selection for a stereo slot, matching the backend's head/wrist
// policy: the overhead uses both eyes (true stereo / depth), a wrist a single
// (left) eye so it costs like a mono camera.
export const defaultEyes = (slot: CameraSlot): StereoEyes => (slot === "overhead" ? "both" : "left")

export const eyesLeft = (e: StereoEyes) => e !== "right"
export const eyesRight = (e: StereoEyes) => e !== "left"

// A per-branch value is "enabled" unless explicitly false; its eye selection
// (stereo only) is the stored eye name, else the slot default.
export const selEnabled = (v: BranchSel | undefined) => v === undefined || v !== false
export const selEyes = (v: BranchSel | undefined, slot: CameraSlot): StereoEyes =>
  v === "both" || v === "left" || v === "right" ? v : defaultEyes(slot)

type BranchMap = Partial<Record<CameraSlot, BranchSel>>

export type CameraProfileBySlot = Partial<Record<CameraSlot, "axol" | "mantis">>

/**
 * Persist exactly what each Cameras control displays. A slot the operator
 * never touched still shows a default (e.g. overhead streams "both"); writing
 * that default explicitly keeps the saved spec matching the panel, so the
 * backend never falls back to a different value (streaming would otherwise
 * default to the recorded eyes). Only edited slots are supplied here: Axol
 * and Mantis share the branch maps, but can assign devices of different kinds
 * to the same logical slot. Explicit values therefore stay byte-for-byte
 * intact unless that slot's control was edited; this function only fills an
 * implicit value for an affected slot whose camera kind is known.
 */
export function materializeCameraSpec(
  spec: CameraSpec,
  devices: CameraDevice[] | null,
  profileBySlot: CameraProfileBySlot
): CameraSpec {
  const kindBySerial = new Map((devices ?? []).map((d) => [String(d.serial), d.kind]))
  const materialize = (map: BranchMap, slot: CameraSlot, stereo: boolean): BranchSel =>
    !selEnabled(map[slot]) ? false : stereo ? selEyes(map[slot], slot) : true
  const outStream: BranchMap = { ...(spec.stream ?? {}) }
  const outRecord: BranchMap = { ...(spec.record ?? {}) }
  for (const key of CAMERA_SLOT_KEYS) {
    const profile = profileBySlot[key]
    if (!profile) continue
    const mappedSerial =
      profile === "mantis" && key !== "overhead"
        ? (spec.mantis_serials?.[key] ?? spec.serials[key])
        : spec.serials[key]
    const kind = kindBySerial.get(mappedSerial.trim())
    if (!kind) continue // unknown kind / unassigned: leave backend defaults
    const stereo = kind === "stereo"
    if (outStream[key] === undefined) {
      outStream[key] = materialize(spec.stream ?? {}, key, stereo)
    }
    if (outRecord[key] === undefined) {
      outRecord[key] = materialize(spec.record ?? {}, key, stereo)
    }
  }
  return {
    serials: {
      overhead: spec.serials.overhead.trim(),
      left_arm: spec.serials.left_arm.trim(),
      right_arm: spec.serials.right_arm.trim(),
    },
    mantis_serials: {
      left_arm: (spec.mantis_serials?.left_arm ?? spec.serials.left_arm).trim(),
      right_arm: (spec.mantis_serials?.right_arm ?? spec.serials.right_arm).trim(),
    },
    stream_resolution: spec.stream_resolution || spec.resolution || "SVGA",
    record_resolution: spec.record_resolution || "SVGA",
    stream: outStream,
    record: outRecord,
  }
}
