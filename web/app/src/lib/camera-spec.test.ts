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
    expect(
      materializeCameraSpec(spec, [
        { serial: 101, model: "ZED X", kind: "stereo" },
        { serial: 202, model: "ZED X One", kind: "mono" },
      ])
    ).toEqual({
      serials: { overhead: "101", left_arm: "202", right_arm: "" },
      stream_resolution: "HD1200",
      record_resolution: "SVGA",
      stream: { overhead: "both", left_arm: false },
      record: { overhead: "right", left_arm: true },
    })
  })
})
