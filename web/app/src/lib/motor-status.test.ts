import { describe, expect, it } from "vitest"

import { statusText } from "./motor-status"

describe("statusText", () => {
  it("gives long fault codes a short tile label and a full name", () => {
    expect(statusText("OUTPUT_SHAFT_CALIBRATION_ERROR")).toEqual({
      short: "Shaft encoder",
      full: "Output shaft encoder calibration fault",
    })
    expect(statusText("MOTOR_ENCODER_CALIBRATION_ERROR").short).toBe("Motor encoder")
  })

  it("keeps every short label tile-sized", () => {
    for (const code of [
      "OK",
      "DISABLED",
      "UNREACHABLE",
      "OVER_VOLTAGE",
      "UNDER_VOLTAGE",
      "OVER_CURRENT",
      "OVER_TEMPERATURE",
      "MOS_OVER_TEMP",
      "ROTOR_OVER_TEMP",
      "LOST_COMM",
      "OVERLOAD",
      "MOTOR_STALL",
      "ENCODER_ERROR",
      "CALIBRATION_ERROR",
      "POWER_OVERRUN",
      "SPEEDING",
      "OUTPUT_SHAFT_CALIBRATION_ERROR",
      "SENSOR_ERROR",
      "MOTOR_ENCODER_CALIBRATION_ERROR",
      "UNKNOWN",
    ]) {
      expect(statusText(code).short.length, code).toBeLessThanOrEqual(13)
    }
  })

  it("makes an unlisted code readable", () => {
    expect(statusText("SOME_NEW_FAULT")).toEqual({
      short: "Some new fault",
      full: "Some new fault",
    })
    expect(statusText(null).short).toBe("–")
  })
})
