import { describe, expect, it } from "vitest"

import {
  defaultEyes,
  eyesLeft,
  eyesRight,
  materializeCameraSpec,
  selEnabled,
  selEyes,
} from "./camera-spec"
import type { CameraSpec } from "./supervisor"

const spec: CameraSpec = {
  serials: { overhead: " 101 ", left_arm: "202", right_arm: "" },
  stream_resolution: "HD1200",
  record_resolution: "SVGA",
  stream: { left_arm: false },
  record: { overhead: "right" },
}

describe("camera specification", () => {
  it("applies slot-specific stereo defaults", () => {
    expect(defaultEyes("overhead")).toBe("both")
    expect(defaultEyes("left_arm")).toBe("left")
    expect(eyesLeft("both")).toBe(true)
    expect(eyesLeft("right")).toBe(false)
    expect(eyesRight("left")).toBe(false)
    expect(selEnabled(undefined)).toBe(true)
    expect(selEnabled(false)).toBe(false)
    expect(selEyes(true, "overhead")).toBe("both")
  })

  it("materializes UI defaults based on detected camera kind", () => {
    const devices = [
      { serial: 101, model: "ZED X", kind: "stereo" as const },
      { serial: 202, model: "ZED X One", kind: "mono" as const },
    ]
    expect(
      materializeCameraSpec(spec, devices, {
        overhead: "axol",
        left_arm: "axol",
        right_arm: "axol",
      })
    ).toEqual({
      serials: { overhead: "101", left_arm: "202", right_arm: "" },
      mantis_serials: { left_arm: "202", right_arm: "" },
      stream_resolution: "HD1200",
      record_resolution: "SVGA",
      stream: { overhead: "both", left_arm: false },
      record: { overhead: "right", left_arm: true },
    })
  })

  it("only fills implicit values for the slots that were edited", () => {
    const devices = [{ serial: 101, model: "ZED X", kind: "stereo" as const }]
    const out = materializeCameraSpec(spec, devices, { left_arm: "axol" })
    // Overhead was not edited, so its implicit stream default stays unset.
    expect(out.stream).toEqual({ left_arm: false })
    expect(out.record).toEqual({ overhead: "right" })
  })

  it("resolves Mantis wrist slots through mantis_serials", () => {
    const devices = [{ serial: 303, model: "ZED X", kind: "stereo" as const }]
    const out = materializeCameraSpec(
      { ...spec, mantis_serials: { left_arm: "303", right_arm: "" } },
      devices,
      { left_arm: "mantis" }
    )
    expect(out.mantis_serials).toEqual({ left_arm: "303", right_arm: "" })
    expect(out.stream).toEqual({ left_arm: false })
    expect(out.record).toEqual({ overhead: "right", left_arm: "left" })
  })
})
