/**
 * Motor status codes as text: a short label that fits a tile, plus the full
 * name for the tooltip and the details dialog. Unlisted codes fall back to
 * the code itself, made readable.
 */
const STATUS_TEXT: Record<string, { short: string; full: string }> = {
  OK: { short: "OK", full: "OK" },
  DISABLED: { short: "Off", full: "Motor off" },
  UNREACHABLE: { short: "No reply", full: "Motor not responding" },
  OVER_VOLTAGE: { short: "Overvoltage", full: "Supply voltage too high" },
  UNDER_VOLTAGE: { short: "Undervoltage", full: "Supply voltage too low" },
  OVER_CURRENT: { short: "Overcurrent", full: "Motor current too high" },
  OVER_TEMPERATURE: { short: "Too hot", full: "Motor too hot" },
  MOS_OVER_TEMP: { short: "Driver hot", full: "Motor driver too hot" },
  ROTOR_OVER_TEMP: { short: "Coil hot", full: "Motor coil too hot" },
  LOST_COMM: { short: "Lost comm", full: "Lost communication" },
  OVERLOAD: { short: "Overload", full: "Motor overloaded" },
  MOTOR_STALL: { short: "Stalled", full: "Motor stalled" },
  ENCODER_ERROR: { short: "Encoder", full: "Encoder fault" },
  CALIBRATION_ERROR: { short: "Calibration", full: "Calibration fault" },
  POWER_OVERRUN: { short: "Power limit", full: "Power limit exceeded" },
  SPEEDING: { short: "Overspeed", full: "Motor too fast" },
  OUTPUT_SHAFT_CALIBRATION_ERROR: {
    short: "Shaft encoder",
    full: "Output shaft encoder calibration fault",
  },
  SENSOR_ERROR: { short: "Sensor", full: "Sensor fault" },
  MOTOR_ENCODER_CALIBRATION_ERROR: {
    short: "Motor encoder",
    full: "Motor encoder calibration fault",
  },
  UNKNOWN: { short: "Fault", full: "Unknown fault" },
}

export function statusText(status: string | null | undefined): { short: string; full: string } {
  if (status == null) return { short: "–", full: "No reading" }
  const known = STATUS_TEXT[status]
  if (known) return known
  const readable = status.replace(/_/g, " ").toLowerCase()
  const text = readable.charAt(0).toUpperCase() + readable.slice(1)
  return { short: text, full: text }
}
