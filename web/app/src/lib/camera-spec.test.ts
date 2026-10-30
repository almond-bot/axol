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
      mantis_serials: { overhead: "", left_arm: "202", right_arm: "" },
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
    expect(out.mantis_serials).toEqual({ overhead: "", left_arm: "303", right_arm: "" })
    expect(out.stream).toEqual({ left_arm: false })
    expect(out.record).toEqual({ overhead: "right", left_arm: "left" })
  })

  it("uses the Mantis head camera kind and saves both stereo eyes by default", () => {
    const out = materializeCameraSpec(
      {
        ...spec,
        mantis_serials: { overhead: " 303 ", left_arm: "202", right_arm: "" },
        stream: {},
        record: {},
      },
      [
        { serial: 101, model: "ZED X One", kind: "mono" },
        { serial: 303, model: "ZED X Mini", kind: "stereo" },
      ],
      { overhead: "mantis" }
    )
    expect(out.serials.overhead).toBe("101")
    expect(out.mantis_serials?.overhead).toBe("303")
    expect(out.stream).toEqual({ overhead: "both" })
    expect(out.record).toEqual({ overhead: "both" })
  })

  it("preserves an explicitly unassigned Mantis head instead of inheriting Axol", () => {
    const out = materializeCameraSpec(
      { ...spec, mantis_serials: { overhead: "" }, stream: {}, record: {} },
      [{ serial: 101, model: "ZED X Mini", kind: "stereo" }],
      { overhead: "mantis" }
    )
    expect(out.mantis_serials?.overhead).toBe("")
    expect(out.stream).toEqual({})
    expect(out.record).toEqual({})
  })

  it("does not save a duplicate inherited head for legacy Mantis wrist assignments", () => {
    const out = materializeCameraSpec(
      { ...spec, mantis_serials: { left_arm: "101", right_arm: "" }, stream: {}, record: {} },
      [{ serial: 101, model: "ZED X Mini", kind: "stereo" }],
      { overhead: "mantis", left_arm: "mantis" }
    )
    expect(out.mantis_serials).toEqual({ overhead: "", left_arm: "101", right_arm: "" })
    expect(out.stream).toEqual({ left_arm: "left" })
    expect(out.record).toEqual({ left_arm: "left" })
  })

  it("can save a partial legacy head-only camera map without inheriting it into Mantis", () => {
    const out = materializeCameraSpec(
      { serials: { overhead: "101" } } as CameraSpec,
      [{ serial: 101, model: "ZED X Mini", kind: "stereo" }],
      { overhead: "mantis" }
    )
    expect(out.serials).toEqual({ overhead: "101", left_arm: "", right_arm: "" })
    expect(out.mantis_serials).toEqual({ overhead: "", left_arm: "", right_arm: "" })
    expect(out.record).toEqual({})
  })
})
