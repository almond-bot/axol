import type { MantisTrackerSource, TrackerCalibrationSnapshot } from "./supervisor"

type PoseConvention = NonNullable<TrackerCalibrationSnapshot["activePoseConvention"]>

/** Whether deleting an override can reveal an approved factory transform. */
export function hasApprovedTrackerFactoryTransform(
  source: MantisTrackerSource,
  convention: PoseConvention | null | undefined
): boolean {
  if (source === "lighthouse") return true
  if (source !== "ultimate") return false
  return convention?.quatOrder === "wxyz" && convention.upAxis === "z"
}
