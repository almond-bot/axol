import type { MotorConfigParam } from "@/lib/telemetry"

/**
 * Presentation for the MyActuator parameter editor: the setup software's
 * panels, labels and selector choices. The backend's parameter table
 * (almond_axol/motor/config.py) is the source of truth for which parameters
 * exist, their units and access; anything it adds that isn't listed here
 * still shows up, under "Other", by its table name.
 */

export interface ParamChoice {
  value: number
  label: string
}

export interface ParamUi {
  label: string
  /** How the value is edited; defaults to a number field. */
  kind?: "number" | "choice" | "toggle" | "display"
  choices?: ParamChoice[]
  hint?: string
}

export const MYACTUATOR_SECTIONS: { title: string; params: string[] }[] = [
  {
    title: "Protect parameters",
    params: [
      "OVER_VOLTAGE",
      "LOW_VOLTAGE",
      "STALL_TIME_LIMIT",
      "EBRAKE_START_DUTY",
      "CURRENT_SAMPLE_RES",
      "EBRAKE_HOLD_DUTY",
      "BRAKE_MODE",
      "ENCODER2_ABNORMAL_VALUE",
      "ENCODER2_ABNORMAL_SPEED",
      "ERROR_DETECTION_LPF_CF",
      "AUTOMATIC_ERROR_RECOVERY",
    ],
  },
  {
    title: "Plan parameters",
    params: [
      "MAX_POSITIVE_POSITION",
      "MIN_NEGATIVE_POSITION",
      "POSITION_PLAN_MAX_ACC",
      "POSITION_PLAN_MAX_DEC",
      "SPEED_PLAN_MAX_ACC",
      "SPEED_PLAN_MAX_DEC",
      "MOTOR_POSITION_ZERO",
      "KT_OUT",
      "MAX_TORQUE",
      "VOLTAGE_SAMPLE_RES",
    ],
  },
  {
    title: "Motor parameters",
    params: [
      "MAX_CURRENT",
      "STALL_CURRENT",
      "SHUTDOWN_TEMP",
      "RESUME_TEMP",
      "MAX_SPEED",
      "NOMINAL_SPEED",
      "RATED_CURRENT",
      "LPF_CF_FOR_CURRENT",
      "LPF_CF_FOR_SPEED",
      "ENABLE_2ND_ENCODER",
      "SECOND_ENCODER_RESOLUTION",
      "ENABLE_ETHERCAT",
      "THERMISTOR",
    ],
  },
  {
    title: "Motor information",
    params: ["MOTOR_NUMBER", "FACTORY_TIME", "REDUCTION_RATIO"],
  },
]

export const PARAM_UI: Record<string, ParamUi> = {
  OVER_VOLTAGE: { label: "Over voltage" },
  LOW_VOLTAGE: {
    label: "Low voltage",
    hint: "The driver sets this to −2 V on enable to suppress the undervoltage fault.",
  },
  STALL_TIME_LIMIT: { label: "Stall time limit" },
  EBRAKE_START_DUTY: { label: "E-brake start duty cycle" },
  CURRENT_SAMPLE_RES: { label: "Current sample resistor" },
  EBRAKE_HOLD_DUTY: { label: "E-brake hold duty cycle" },
  BRAKE_MODE: {
    label: "Brake mode",
    kind: "choice",
    choices: [
      { value: 0, label: "E-Brake" },
      { value: 1, label: "Braking resistor" },
    ],
  },
  ENCODER2_ABNORMAL_VALUE: { label: "Encoder2 abnormal value" },
  ENCODER2_ABNORMAL_SPEED: { label: "Encoder2 abnormal speed" },
  ERROR_DETECTION_LPF_CF: { label: "Error detection LPF cutoff" },
  AUTOMATIC_ERROR_RECOVERY: { label: "Automatic error recovery", kind: "toggle" },
  MAX_POSITIVE_POSITION: { label: "Max positive position" },
  MIN_NEGATIVE_POSITION: { label: "Min negative position" },
  POSITION_PLAN_MAX_ACC: { label: "Position plan max acceleration" },
  POSITION_PLAN_MAX_DEC: { label: "Position plan max deceleration" },
  SPEED_PLAN_MAX_ACC: { label: "Speed plan max acceleration" },
  SPEED_PLAN_MAX_DEC: { label: "Speed plan max deceleration" },
  MOTOR_POSITION_ZERO: {
    label: "Motor position zero",
    hint: "Raw encoder zero. Use Set zero position to re-zero a joint.",
  },
  KT_OUT: { label: "KT_OUT" },
  MAX_TORQUE: { label: "Max torque" },
  VOLTAGE_SAMPLE_RES: { label: "Voltage sample resistor" },
  MAX_CURRENT: { label: "Max current" },
  STALL_CURRENT: { label: "Stall current" },
  SHUTDOWN_TEMP: { label: "Shutdown temperature" },
  RESUME_TEMP: { label: "Resume temperature" },
  MAX_SPEED: { label: "Max speed" },
  NOMINAL_SPEED: { label: "Nominal speed" },
  RATED_CURRENT: { label: "Rated current" },
  LPF_CF_FOR_CURRENT: { label: "LPF cutoff for current data" },
  LPF_CF_FOR_SPEED: { label: "LPF cutoff for speed data" },
  ENABLE_2ND_ENCODER: {
    label: "Enable 2nd encoder",
    kind: "choice",
    choices: [
      { value: 0, label: "Disabled" },
      { value: 1, label: "Encoder1 (262144)" },
      { value: 2, label: "Encoder2 (16384)" },
      { value: 3, label: "Encoder3 (131072)" },
    ],
  },
  SECOND_ENCODER_RESOLUTION: {
    label: "2nd encoder resolution",
    kind: "display",
    hint: "Set together with the 2nd encoder mode.",
  },
  ENABLE_ETHERCAT: { label: "Enable EtherCAT", kind: "toggle" },
  THERMISTOR: {
    label: "Thermistor",
    kind: "choice",
    choices: [
      { value: 0, label: "Thermistor1" },
      { value: 1, label: "Thermistor2" },
    ],
  },
  MOTOR_NUMBER: { label: "Motor number" },
  FACTORY_TIME: { label: "Factory time" },
  REDUCTION_RATIO: { label: "Reduction ratio" },
}

const UNIT_LABELS: Record<string, string> = {
  C: "°C",
  deg: "°",
  kOhm: "kΩ",
  mOhm: "mΩ",
  rpm: "RPM",
}

export function unitLabel(unit: string): string {
  return UNIT_LABELS[unit] ?? unit
}

export function paramUi(name: string): ParamUi {
  return PARAM_UI[name] ?? { label: name }
}

/** Editable text for a value: whole numbers without decimals, others trimmed. */
export function formatParamValue(value: number | null): string {
  if (value == null) return ""
  if (Number.isInteger(value)) return String(value)
  return String(Number(value.toFixed(4)))
}

/** A draft field's number, or null unless it is a finite number. */
export function parseParamInput(text: string): number | null {
  if (text.trim() === "") return null
  const value = Number(text)
  return Number.isFinite(value) ? value : null
}

/** The table's parameters in panel order; unlisted ones land in "Other". */
export function groupParams(
  params: MotorConfigParam[]
): { title: string; params: MotorConfigParam[] }[] {
  const byName = new Map(params.map((p) => [p.name, p]))
  const placed = new Set<string>()
  const sections = MYACTUATOR_SECTIONS.map((section) => ({
    title: section.title,
    params: section.params.flatMap((name) => {
      const param = byName.get(name)
      if (!param) return []
      placed.add(name)
      return [param]
    }),
  })).filter((section) => section.params.length > 0)
  const other = params.filter((p) => !placed.has(p.name))
  if (other.length > 0) sections.push({ title: "Other", params: other })
  return sections
}
